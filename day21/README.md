# 第21天 智能客服模型微调

## 一、环境建立

```bash
# 创建虚拟环境
conda create -n day21 python=3.12 -y
# 激活虚拟环境
conda activate day21
# 安装依赖库
pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple
# 安装Pytorch
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
# 校验PyTorch
python -c "import torch; print(torch.cuda.is_available())"
```

## 二、下载模型

```bash
python model_download2.py --repo_id Qwen/Qwen3-1.7B
```

## 三、数据说明

训练数据已准备完成：
- `data/train_qa.jsonl` — 16,051 条问答对
- `data/val_qa.jsonl` — 1,422 条验证集问答对
- `data/test_qa.jsonl` — 1,359 条测试集问答对

每条数据格式：
```json
{"question": "现在是否没有纸质发票了。", "answer": "是的，都是电子发票了。", "intent": "补发票"}
```

## 四、测试原始模型

```bash
# 启动API Server
python api_server.py --model-name models/Qwen/Qwen3-1.7B
# 开启WebUI测试（另开窗口）
conda activate day21
python kefu_chat.py
```

## 五、模型训练

```bash
# 训练（包含验证集评估）
python kefu_train_trl.py
```

训练完成后会在 `./qwen3_kefu_sft_output/` 下生成 checkpoint，Loss曲线保存在 `./loss_plots/`。

## 六、测试训练结果

```bash
# 使用训练好的模型启动服务（将 <checkpoint-xxx> 替换为实际文件夹名）
python api_server.py --model-name ./qwen3_kefu_sft_output/checkpoint-xxx
# 开启WebUI测试（另开窗口）
conda activate day21
python kefu_chat.py
```
