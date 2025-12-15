# 第6天 大模型企业级部署技术

## 一、建立虚拟环境

```powershell
# 1、创建虚拟环境
conda create -n day06 python=3.12 -y
# 2、激活虚拟环境
conda activate day06
```

## 二、安装vLLM For Linux（不实践）

```powershell
pip install vllm -i https://pypi.mirrors.ustc.edu.cn/simple
```

## 三、安装vLLM For Windows

### 1、安装Triton

```powershell
# 下载链接
https://huggingface.co/madbuda/triton-windows-builds/resolve/main/triton-3.0.0-cp312-cp312-win_amd64.whl
# 安装
pip install triton-3.0.0-cp312-cp312-win_amd64.whl -i https://pypi.mirrors.ustc.edu.cn/simple
```

### 2、安装vLLM

```powershell
# 下载链接
https://github.com/SystemPanic/vllm-windows/releases/download/v0.10.2/vllm-0.10.2+cu124-cp312-cp312-win_amd64.whl
# 安装vLLM
pip install vllm-0.10.2+cu124-cp312-cp312-win_amd64.whl -i https://pypi.mirrors.ustc.edu.cn/simple
# 重装PyTorch
pip install torch==2.7.1+cu126 torchaudio==2.7.1+cu126 torchvision==0.22.1+cu126 --index-url https://download.pytorch.org/whl/cu126
# 校验PyTorch是否正确安装
python -c "import torch; print(torch.cuda.is_available())"
```

## 四、下载模型

```powershell
# 获取下载脚本
https://aliendao.cn/model_download2.py
# 下载模型权重
python model_download2.py --repo_id deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```

## 五、运行模型服务

```powershell
# 禁用TORCHDYNAMO_DISABLE
set TORCHDYNAMO_DISABLE=1
# 启动vLLM服务
vllm serve ./models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --served-model-name=deepseek --dtype=half --gpu_memory_utilization=0.8 --max_model_len=1024
```

## 六、测试vLLM服务

```powershell
# 1、激活虚拟环境
conda activate day06
# 2、安装Gradio
pip install gradio==6.0.2 -i https://pypi.mirrors.ustc.edu.cn/simple
# 3、运行Chatbot测试
python chatbot.py
```

