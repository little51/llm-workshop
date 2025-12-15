import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from typing import List, Dict, Tuple
import json
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm


class GRPOTrainerWithEvaluation:
    def __init__(self, model_name="./models/Qwen/Qwen3-0.6B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        # 保存原始模型用于对比
        self.original_model = AutoModelForCausalLM.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.original_model.to(self.device)
        self.original_model.eval()  # 原始模型设为评估模式
        # 评估指标记录
        self.training_history = {
            'epoch_losses': [],
            'response_scores': [],
            'advantages': [],
            'evaluation_metrics': []
        }

    def generate_responses(self, prompt: str, num_responses: int = 4, model=None) -> tuple:
        """为单个prompt生成多个响应"""
        if model is None:
            model = self.model

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)

        all_responses = []
        all_log_probs = []
        model.eval() if model == self.original_model else model.train()
        with torch.set_grad_enabled(model.training):
            for _ in range(num_responses):
                # 生成响应
                outputs = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    return_dict_in_generate=True,
                    output_scores=True
                )

                # 获取生成的tokens
                generated_tokens = outputs.sequences[:,
                                                     inputs.input_ids.shape[1]:]

                # 计算log probabilities
                scores = outputs.scores  # 每个生成步骤的logits
                log_probs = []
                for step, score in enumerate(scores):
                    logits = score
                    log_prob = F.log_softmax(logits, dim=-1)
                    # 获取实际生成的token的概率
                    next_token = generated_tokens[:, step]
                    token_log_prob = log_prob[torch.arange(
                        log_prob.size(0)), next_token]
                    log_probs.append(token_log_prob)
                # 整个序列的对数概率
                seq_log_prob = torch.stack(log_probs).sum(dim=0)
                # 解码响应文本
                response_text = self.tokenizer.decode(
                    generated_tokens[0], skip_special_tokens=True)
                all_responses.append(response_text)
                all_log_probs.append(seq_log_prob.item())
        return all_responses, all_log_probs, inputs.input_ids.shape[1]

    def evaluate_responses(self, prompt: str, responses: List[str]) -> Tuple[List[float], Dict]:
        """
        评估响应质量 - 不知道"正确答案"，只能相对比较
        返回：分数列表和详细的评估指标
        """
        scores = []
        detailed_metrics = []
        for response in responses:
            metrics = {}
            # 1. 响应长度（适中最好）
            response_len = len(response.split())
            metrics['length'] = response_len
            if response_len < 10:
                length_score = -1.0  # 太短的响应不好
            elif response_len > 200:
                length_score = -0.5  # 太长的响应可能啰嗦
            else:
                length_score = min(response_len / 50, 1.0)  # 适中的长度更好
            metrics['length_score'] = length_score
            # 2. 响应相关性（简单关键词匹配）
            prompt_words = set(prompt.lower().split())
            response_words = set(response.lower().split())
            common_words = prompt_words.intersection(response_words)
            relevance = 0
            if len(prompt_words) > 0:
                relevance = len(common_words) / len(prompt_words)
            metrics['relevance'] = relevance
            relevance_score = relevance * 2.0
            metrics['relevance_score'] = relevance_score
            # 3. 响应结构质量
            # 检查是否有完整句子
            has_complete_sentence = ('.' in response or '。' in response)
            metrics['has_complete_sentence'] = has_complete_sentence
            structure_score = 0.5 if has_complete_sentence else 0.0
            metrics['structure_score'] = structure_score
            # 4. 词汇多样性
            words = response.split()
            unique_words = set(words)
            if len(words) > 0:
                diversity = len(unique_words) / len(words)
            else:
                diversity = 0
            metrics['diversity'] = diversity
            diversity_score = diversity * 1.0
            metrics['diversity_score'] = diversity_score
            # 5. 具体性（检查数字、具体例子等）
            specificity_indicators = 0
            if any(char.isdigit() for char in response):
                specificity_indicators += 1
            if '例如' in response or '比如' in response or '举例' in response:
                specificity_indicators += 1
            if '具体' in response or '详细' in response:
                specificity_indicators += 1
            metrics['specificity_indicators'] = specificity_indicators
            specificity_score = min(specificity_indicators * 0.3, 1.0)
            metrics['specificity_score'] = specificity_score
            # 总分
            total_score = (length_score + relevance_score + structure_score +
                           diversity_score + specificity_score)
            metrics['total_score'] = total_score
            scores.append(total_score)
            detailed_metrics.append(metrics)
        return scores, detailed_metrics

    def compute_advantages(self, scores: List[float]) -> List[float]:
        """计算相对优势（relative advantage）"""
        scores_array = np.array(scores)
        mean_score = np.mean(scores_array)
        std_score = np.std(scores_array) + 1e-8
        # 标准化分数
        normalized_scores = (scores_array - mean_score) / std_score
        return normalized_scores.tolist()

    def grpo_loss(self, log_probs: List[float], advantages: List[float]) -> torch.Tensor:
        """计算GRPO损失"""
        losses = []
        for log_prob, advantage in zip(log_probs, advantages):
            # 核心思想：让优势更大的响应更可能被生成
            loss = -advantage * log_prob
            losses.append(loss)
        return torch.mean(torch.stack(losses))

    def train_step(self, prompt: str, optimizer, num_responses: int = 4):
        """单个训练步骤"""
        self.model.train()
        # 1. 生成多个响应
        responses, log_probs, prompt_length = self.generate_responses(
            prompt, num_responses)
        # 2. 评估响应（不知道正确答案，只能相对比较）
        scores, detailed_metrics = self.evaluate_responses(prompt, responses)
        # 3. 计算相对优势
        advantages = self.compute_advantages(scores)
        # 记录训练数据
        self.training_history['response_scores'].extend(scores)
        self.training_history['advantages'].extend(advantages)
        # 4. 计算损失
        log_probs_tensor = torch.tensor(
            log_probs, device=self.device, requires_grad=True)
        advantages_tensor = torch.tensor(advantages, device=self.device)
        loss = self.grpo_loss(log_probs_tensor, advantages_tensor)
        # 5. 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()
        return loss.item(), responses, scores, detailed_metrics, advantages

    def evaluate_model_comparison(self, test_prompts: List[str], num_responses: int = 3) -> Dict:
        """
        对比评估：原始模型 vs 强化后模型
        """
        print("\n" + "="*60)
        print("模型对比评估")
        print("="*60)

        comparison_results = {
            'original_model': [],
            'fine_tuned_model': [],
            'improvement': []
        }

        for prompt in test_prompts:
            print(f"\n测试Prompt: {prompt[:80]}...")
            # 原始模型生成
            print("原始模型生成:")
            orig_responses, orig_log_probs, _ = self.generate_responses(
                prompt, num_responses, model=self.original_model
            )
            orig_scores, orig_metrics = self.evaluate_responses(
                prompt, orig_responses)
            avg_orig_score = np.mean(orig_scores)
            # 强化后模型生成
            print("强化后模型生成:")
            ft_responses, ft_log_probs, _ = self.generate_responses(
                prompt, num_responses, model=self.model
            )
            ft_scores, ft_metrics = self.evaluate_responses(
                prompt, ft_responses)
            avg_ft_score = np.mean(ft_scores)
            # 计算改进
            improvement = ((avg_ft_score - avg_orig_score) /
                           (abs(avg_orig_score) + 1e-8)) * 100
            # 显示结果
            print(f"原始模型平均分数: {avg_orig_score:.3f}")
            for i, (resp, score) in enumerate(zip(orig_responses, orig_scores)):
                print(f"  响应{i+1} ({score:.2f}): {resp[:80]}...")
            print(f"强化后模型平均分数: {avg_ft_score:.3f}")
            for i, (resp, score) in enumerate(zip(ft_responses, ft_scores)):
                print(f"  响应{i+1} ({score:.2f}): {resp[:80]}...")
            print(f"改进幅度: {improvement:.1f}%")
            # 记录结果
            comparison_results['original_model'].append({
                'prompt': prompt,
                'responses': orig_responses,
                'scores': orig_scores,
                'avg_score': avg_orig_score,
                'metrics': orig_metrics
            })
            comparison_results['fine_tuned_model'].append({
                'prompt': prompt,
                'responses': ft_responses,
                'scores': ft_scores,
                'avg_score': avg_ft_score,
                'metrics': ft_metrics
            })
            comparison_results['improvement'].append(improvement)
        # 计算总体统计
        avg_orig = np.mean([r['avg_score']
                           for r in comparison_results['original_model']])
        avg_ft = np.mean([r['avg_score']
                         for r in comparison_results['fine_tuned_model']])
        avg_improvement = np.mean(comparison_results['improvement'])
        print(f"\n{'='*60}")
        print(f"总体统计:")
        print(f"原始模型平均分: {avg_orig:.3f}")
        print(f"强化后模型平均分: {avg_ft:.3f}")
        print(f"平均改进幅度: {avg_improvement:.1f}%")
        print("="*60)
        comparison_results['summary'] = {
            'avg_original_score': avg_orig,
            'avg_finetuned_score': avg_ft,
            'avg_improvement_percent': avg_improvement
        }
        return comparison_results

    def analyze_training_progress(self):
        """分析训练进度"""
        if not self.training_history['epoch_losses']:
            print("暂无训练数据可分析")
            return
        # 绘制损失曲线
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot(self.training_history['epoch_losses'])
        plt.title('Training Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        if self.training_history['response_scores']:
            plt.subplot(1, 3, 2)
            scores = self.training_history['response_scores']
            # 计算移动平均
            window_size = min(50, len(scores))
            moving_avg = np.convolve(scores, np.ones(
                window_size)/window_size, mode='valid')
            plt.plot(scores, alpha=0.3, label='Raw scores')
            plt.plot(range(window_size-1, len(scores)), moving_avg, 'r-',
                     label=f'{window_size}-step moving avg')
            plt.title('Response Scores during Training')
            plt.xlabel('Training Step')
            plt.ylabel('Score')
            plt.legend()
            plt.grid(True)

        if self.training_history['advantages']:
            plt.subplot(1, 3, 3)
            advantages = self.training_history['advantages']
            plt.hist(advantages, bins=30, alpha=0.7)
            plt.title('Distribution of Advantages')
            plt.xlabel('Advantage')
            plt.ylabel('Frequency')
            plt.grid(True)

        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
        plt.show()
        # 打印统计信息
        print("\n" + "="*60)
        print("训练进度分析")
        print("="*60)
        if self.training_history['response_scores']:
            scores = self.training_history['response_scores']
            print(f"响应分数统计:")
            print(f"  平均分: {np.mean(scores):.3f}")
            print(f"  标准差: {np.std(scores):.3f}")
            print(f"  最小值: {np.min(scores):.3f}")
            print(f"  最大值: {np.max(scores):.3f}")

        if self.training_history['advantages']:
            advantages = self.training_history['advantages']
            print(f"\n优势值统计:")
            print(f"  平均优势: {np.mean(advantages):.3f}")
            print(
                f"  优势>0的比例: {np.sum(np.array(advantages) > 0) / len(advantages):.1%}")

    def save_evaluation_report(self, comparison_results: Dict, filename: str = "grpo_evaluation_report.json"):
        """保存评估报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_name': "Qwen3-0.6B-GRPO",
            'training_summary': {
                'num_epochs': len(self.training_history['epoch_losses']),
                'final_loss': self.training_history['epoch_losses'][-1] if self.training_history['epoch_losses'] else None,
            },
            'comparison_results': comparison_results,
            'training_history_summary': {
                'avg_loss': np.mean(self.training_history['epoch_losses']) if self.training_history['epoch_losses'] else None,
                'avg_score': np.mean(self.training_history['response_scores']) if self.training_history['response_scores'] else None,
            }
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"评估报告已保存到: {filename}")

# 使用示例
def main():
    # 初始化训练器
    trainer = GRPOTrainerWithEvaluation("./models/Qwen/Qwen3-0.6B")
    # 定义优化器
    optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-6)
    # 训练数据
    training_prompts = [
        "请解释量子计算的基本原理",
        "写一个关于人工智能的短故事",
        "如何提高深度学习模型的泛化能力？",
        "描述区大模型技术的三个主要应用",
        "用简单的语言解释相对论",
        "写一首关于科技的现代诗",
        "分析可再生能源的未来发展趋势",
        "如何培养创造性思维？",
        "描述一个可持续发展的城市应该具备哪些特征",
        "解释什么是元宇宙及其潜在影响"
    ]
    # 测试数据（用于评估）
    test_prompts = [
        "什么是机器学习？",
        "如何学习编程？",
        "描述云计算的优势",
        "人工智能有哪些伦理问题？",
        "大数据在商业中有哪些应用？"
    ]

    print("开始训练前的基准评估...")
    baseline_comparison = trainer.evaluate_model_comparison(
        test_prompts[:2], num_responses=2)
    # 训练循环
    num_epochs = 3
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")
        epoch_losses = []
        epoch_scores = []
        for i, prompt in enumerate(tqdm(training_prompts, desc=f"Epoch {epoch+1}")):
            loss, responses, scores, detailed_metrics, advantages = trainer.train_step(
                prompt, optimizer, num_responses=4
            )
            epoch_losses.append(loss)
            epoch_scores.extend(scores)
            # 每3个prompt显示一次详细结果
            if i % 3 == 0:
                print(f"\nPrompt: {prompt[:60]}...")
                best_idx = np.argmax(scores)
                print(
                    f"Best response (score: {scores[best_idx]:.2f}): {responses[best_idx][:100]}...")
        avg_epoch_loss = np.mean(epoch_losses)
        avg_epoch_score = np.mean(epoch_scores)
        trainer.training_history['epoch_losses'].append(avg_epoch_loss)
        print(f"\nEpoch {epoch + 1} 统计:")
        print(f"  平均损失: {avg_epoch_loss:.4f}")
        print(f"  平均响应分数: {avg_epoch_score:.3f}")
        # 每个epoch后进行简单评估
        if epoch % 1 == 0:  # 每个epoch都评估
            print(f"\nEpoch {epoch + 1} 后模型评估...")
            _ = trainer.evaluate_model_comparison(
                test_prompts[:1], num_responses=2)

    # 训练完成后的全面评估
    print("\n" + "="*60)
    print("训练完成，开始全面评估")
    print("="*60)
    # 1. 分析训练进度
    trainer.analyze_training_progress()
    # 2. 对比评估
    final_comparison = trainer.evaluate_model_comparison(
        test_prompts, num_responses=3)
    # 3. 保存评估报告
    trainer.save_evaluation_report(final_comparison)
    # 4. 保存模型
    trainer.model.save_pretrained("./qwen3_0.6b_grpo_finetuned")
    trainer.tokenizer.save_pretrained("./qwen3_0.6b_grpo_finetuned")
    print("\n模型已保存到: ./qwen3_0.6b_grpo_finetuned")
    # 5. 显示改进总结
    print("\n" + "="*60)
    print("强化训练效果总结")
    print("="*60)
    baseline_avg = np.mean([r['avg_score']
                           for r in baseline_comparison['original_model']])
    final_avg = np.mean([r['avg_score']
                        for r in final_comparison['fine_tuned_model']])
    improvement = ((final_avg - baseline_avg) /
                   (abs(baseline_avg) + 1e-8)) * 100

    print(f"训练前基准分数: {baseline_avg:.3f}")
    print(f"训练后最终分数: {final_avg:.3f}")
    print(f"总体改进幅度: {improvement:.1f}%")
    # 显示具体指标的改进
    print("\n具体指标改进:")
    print("指标项            | 改进情况")
    print("-" * 40)
    # 计算各个指标的改进
    metric_names = ['length_score', 'relevance_score', 'structure_score',
                    'diversity_score', 'specificity_score']
    for metric in metric_names:
        baseline_vals = []
        final_vals = []
        for baseline_resp in baseline_comparison['original_model']:
            for resp_metrics in baseline_resp['metrics']:
                baseline_vals.append(resp_metrics[metric])
        for final_resp in final_comparison['fine_tuned_model']:
            for resp_metrics in final_resp['metrics']:
                final_vals.append(resp_metrics[metric])
        if baseline_vals and final_vals:
            baseline_mean = np.mean(baseline_vals)
            final_mean = np.mean(final_vals)
            metric_improvement = (
                (final_mean - baseline_mean) / (abs(baseline_mean) + 1e-8)) * 100
            print(f"{metric:<15} | {metric_improvement:>6.1f}%")

if __name__ == "__main__":
    main()
