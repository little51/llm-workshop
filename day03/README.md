# 第3天 大模型部署（一）

## 一、安装Xinference

```powershell
# 1、创建虚拟环境
conda create -n day03 python=3.12 -y
# 2、激活虚拟环境
conda activate day03
# 3、安装PyTorch
pip install xinference[all]==1.13.0 gradio==5.49.1 -i https://pypi.mirrors.ustc.edu.cn/simple
# 4、校验PyTorch是否正确安装
python -c "import torch; print(torch.cuda.is_available())"
# 5、如PyTorch安装不正确，重装PyTorch
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
# 6、重装后再次校验PyTorch是否正确安装
python -c "import torch; print(torch.cuda.is_available())"
```

## 二、运行Xinference

```powershell
# 激活虚拟环境
conda activate day03
# 运行
xinference-local
# 访问
http://127.0.0.1:9997
```

## 三、装载与测试模型

### 1、WebUI装载

```powershell
# 测试qwen0.6b模型
# 模型使用显存的估算方法
模型参数量 * 2
如 0.6B * 2 = 1.2G
    4B * 2 = 8Gs
```

### 2、命令行装载

```powershell
# 1、再开另一个命令窗口
# 2、激活虚拟环境
conda activate day03
# 3、运行命令
xinference launch --model-name qwen3 --model-type LLM --model-engine Transformers --model-format pytorch --size-in-billions 0_6 --quantization none --n-gpu auto --replica 1 --download_hub modelscope --enable_thinking true --reasoning_content false
# 4、访问
http://127.0.0.1:9997/qwen3/
```

## 四、最简ChatBot

```powershell
# 1、再开另一个命令窗口
# 2、激活虚拟环境
conda activate day03
# 3、升级Gradio（本例的Gradio与Xinference用的Gradio版本有冲突）
pip install gradio==6.0.1 -i https://pypi.mirrors.ustc.edu.cn/simple
# 4、运行ChatBot
python chatbot.py
# 5、访问ChatBot
http://localhost:7860
```

### 思考题：

1、让本机以外的机器访问ChatBot该如何写？

2、如果更换模型该如何改？

3、多轮会话是如何实现的？

4、提示词越来越长该如何解决？