from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)

import torch
from trl import SFTTrainer

# 1. 模型和分词器加载
model_name = "models/Qwen/Qwen3-1.7B"

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    use_cache=False,
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    padding_side="right",
)

# 设置pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. 配置LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

# 3. 启用梯度检查点
model.gradient_checkpointing_enable()

# 4. 加载和预处理数据集（使用本地JSONL文件，含验证集）
dataset = load_dataset(
    "json",
    data_files={
        "train": "data/train_qa.jsonl",
        "val": "data/val_qa.jsonl",
    },
)

def format_dataset(example):
    """将问答对转换为Qwen3的对话格式"""
    messages = [
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

# 应用格式转换
dataset = dataset.map(format_dataset)

print(f"训练集大小: {len(dataset['train'])}")
print(f"验证集大小: {len(dataset['val'])}")

# 5. 配置训练参数
training_args = TrainingArguments(
    output_dir="./qwen3_kefu_sft_output",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    eval_strategy="steps",
    eval_steps=500,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    optim="adamw_8bit",
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    gradient_checkpointing=True,
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    report_to=[],
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# 6. 创建训练器（同时指定训练集和验证集）
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["val"],
    args=training_args,
)

# 7. 打印可训练参数
model.print_trainable_parameters()

# 8. 开始训练
trainer.train()

# 9. 保存模型
trainer.save_model("./qwen3_kefu_finetuned")

# 10. 绘制Loss曲线
import matplotlib.pyplot as plt
import json
import os

def plot_training_loss(log_history, output_dir="./loss_plots"):
    """从训练历史中绘制Loss曲线（同时包含训练和验证loss）"""
    os.makedirs(output_dir, exist_ok=True)

    # 提取loss数据
    steps = []
    losses = []
    eval_steps = []
    eval_losses = []

    for log in log_history:
        if "loss" in log and "eval_loss" not in log:
            steps.append(log.get("step", 0))
            losses.append(log["loss"])
        if "eval_loss" in log:
            eval_steps.append(log.get("step", 0))
            eval_losses.append(log["eval_loss"])

    # 绘制图表
    plt.figure(figsize=(10, 6))
    if losses:
        plt.plot(steps, losses, 'b-', linewidth=2, label='Train Loss')
    if eval_losses:
        plt.plot(eval_steps, eval_losses, 'r--', linewidth=2, label='Eval Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training and Evaluation Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 保存图片
    plot_path = os.path.join(output_dir, "training_loss.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')

    # 保存数据到文件
    data = {
        "steps": steps,
        "losses": losses,
        "eval_steps": eval_steps,
        "eval_losses": eval_losses,
    }
    with open(os.path.join(output_dir, "loss_data.json"), "w") as f:
        json.dump(data, f, indent=2)

    print(f"Loss曲线已保存到: {plot_path}")
    print(f"Loss数据已保存到: {os.path.join(output_dir, 'loss_data.json')}")
    if losses:
        print(f"最终训练Loss: {losses[-1]:.4f}")
    if eval_losses:
        print(f"最终验证Loss: {eval_losses[-1]:.4f}")

plot_training_loss(trainer.state.log_history, output_dir="./loss_plots")
