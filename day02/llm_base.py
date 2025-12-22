from transformers import AutoModelForCausalLM, AutoTokenizer

# 1、模型名称或路径
model_name = "models/Qwen/Qwen3-0.6B"

# 2、装载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto"
)

# 3、大模型的思考模式示例
prompt = "你好，介绍一下你自己。"

# 4、组织对话消息
messages = [
    {"role": "user", "content": prompt}
]

# 5、应用聊天模板，启用思考模式
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # 启用思考模式
)

# 6、编码输入文本(转为张量并移动到模型设备上
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 7、生成响应，设置max_new_tokens为32768以支持长输出
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)

# 8、提取生成的输出ID
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# 9、解析思考内容和最终回答
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

# 10、打印思考内容和最终回答
print("========== 思考过程 ========\n", thinking_content)
print("\n\n========== 最终答案 ========\n", content)
