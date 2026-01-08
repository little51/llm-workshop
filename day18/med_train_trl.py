from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType
)
from datasets import load_dataset
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
    use_cache=False,  # 用于梯度检查点
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

# 4. 加载和预处理数据集
dataset = load_dataset("my_datasets/michaelwzhu/ShenNong_TCM_Dataset")

def format_dataset(example):
    """将数据转换为Qwen3的对话格式"""
    messages = [
        {"role": "user", "content": example["query"]},
        {"role": "assistant", "content": example["response"]},
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

# 应用格式转换
dataset = dataset.map(format_dataset)

# 5. 配置训练参数
training_args = TrainingArguments(
    output_dir="./qwen3_tcm_sft_output_standard",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    optim="adamw_8bit",
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    gradient_checkpointing=True,
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    report_to=[],  # 禁用wandb等记录
    save_total_limit=3
)

# 6. 创建训练器
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"] if "train" in dataset else dataset,
    args=training_args    
)

# 7. 打印可训练参数
model.print_trainable_parameters()

# 8. 开始训练
trainer.train()
#checkpoint_path = "./qwen3_tcm_sft_output_standard/checkpoint-43500"
#trainer.train(resume_from_checkpoint = checkpoint_path)

# 9. 保存模型
trainer.save_model("./qwen3_tcm_finetuned_standard")

# 在训练完成后添加以下代码
import matplotlib.pyplot as plt
import json
import os

def plot_training_loss(log_history, output_dir="./loss_plots"):
    """从训练历史中绘制Loss曲线"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取loss数据
    steps = []
    losses = []
    
    for log in trainer.state.log_history:
        if "loss" in log:
            steps.append(log.get("step", 0))
            losses.append(log["loss"])
    
    if not losses:
        print("没有找到loss数据")
        return
    
    # 绘制图表
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, 'b-', linewidth=2)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True, alpha=0.3)
    
    # 保存图片
    plot_path = os.path.join(output_dir, "training_loss.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # 保存数据到文件
    data = {
        "steps": steps,
        "losses": losses
    }
    with open(os.path.join(output_dir, "loss_data.json"), "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Loss曲线已保存到: {plot_path}")
    print(f"Loss数据已保存到: {os.path.join(output_dir, 'loss_data.json')}")

plot_training_loss(trainer.state.log_history, output_dir="./loss_plots")