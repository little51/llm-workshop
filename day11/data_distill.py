import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 1. 加载数据集
print("正在加载数据集...")
dataset = load_dataset("./datasets/AI-MO/NuminaMath-TIR", split="train")

# 2. 加载Qwen模型
print("正在加载Qwen模型...")
model_name = "./models/Qwen/Qwen2.5-Math-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16
).to("cuda:0")

# 3. 创建蒸馏提示模板
def create_prompt(question):
    return f"""请解答以下数学问题，并给出详细的思考过程：
        问题：{question}
        """

# 4. 生成蒸馏数据
def distill_data(num_samples=50, output_file="distilled_data.jsonl"):
    distilled_results = []
    for i, example in enumerate(dataset):
        if i >= num_samples:
            break
        try:
            question = example.get("problem", "")
            if not question:
                continue
            # 创建提示
            prompt = create_prompt(question)
            # 编码输入
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            # 生成回答
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            # 解码输出
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 保存结果
            result = {
                "question": question,
                "answer": response
            }
            distilled_results.append(result)
            # 打印进度
            print(f"已处理 {i+1}/{num_samples} 个样本")
            print("-" * 50)

        except Exception as e:
            print(f"处理样本 {i} 时出错: {e}")
            continue

    # 保存到文件
    with open(output_file, "w", encoding="utf-8") as f:
        for result in distilled_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"数据已保存到 {output_file}")


if __name__ == "__main__":
    distill_data(num_samples=10)
