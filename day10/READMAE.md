# 第10天 大模型强化学习

## 一、环境建立

```powershell
# 创建虚拟环境
conda create -n day10 python=3.12 -y
# 激活虚拟环境
conda activate day10
# 安装依赖库
pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple
# 重装Pytorch(Windows)
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
```

## 二、模型下载

### 方法1：从Hugginface或镜像下载

```powershell
# 权重文件下载地址
https://hf-mirror.com/Qwen/Qwen3-0.6B/tree/main
# 权重下载（下载的10个文件放到Qwen/Qwen2.5-0.5B目录下）
```

### 方法2：从aliendao.cn下载

```powershell
# 获取下载脚本
https://aliendao.cn/model_download2.py
# 下载模型权重
python model_download2.py --repo_id Qwen/Qwen3-0.6B
```

## 三、强化学习过程

```powershell
python train_grpo.py
```