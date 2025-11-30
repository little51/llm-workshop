# 第6天 大模型部署（三）

## 一、建立虚拟环境

```powershell
# 1、创建虚拟环境
conda create -n day06 python=3.12 -y
# 2、激活虚拟环境
conda activate day06
```

## 二、安装Triton

```powershell
# 下载链接
https://huggingface.co/madbuda/triton-windows-builds/resolve/main/triton-3.0.0-cp312-cp312-win_amd64.whl
# 安装
pip install triton-3.0.0-cp312-cp312-win_amd64.whl -i https://pypi.mirrors.ustc.edu.cn/simple
```

## 三、安装vLLM For Windows

```powershell
# 下载链接
https://github.com/SystemPanic/vllm-windows/releases/download/v0.8.5/vllm-0.8.5+cu124-cp312-cp312-win_amd64.whl
# 安装vLLM
pip install vllm-0.8.5+cu124-cp312-cp312-win_amd64.whl -i https://pypi.mirrors.ustc.edu.cn/simple
```

## 四、安装Xformer

```powershell
pip install xformers==0.0.27 -i https://pypi.mirrors.ustc.edu.cn/simple
```

## 五、重装PyTorch

```powershell
# 重装PyTorch
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
# 校验PyTorch是否正确安装
python -c "import torch; print(torch.cuda.is_available())"
```

## 六、下载模型

```powershell
# 获取下载脚本
https://aliendao.cn/model_download2.py
# 下载模型权重
python model_download2.py --repo_id Qwen/Qwen3-0.6B
```

## 七、运行模型服务

```powershell
vllm serve ./models/Qwen/Qwen3-0.6B --enable-reasoning --reasoning-parser deepseek_r1 --dtype=half
```

