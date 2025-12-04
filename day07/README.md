# 第7课 多模态模型部署与应用

## 一、Qwen-VL模型

### 1、环境建立

```powershell
# 创建虚拟环境
conda create -n day07 python=3.12 -y
# 激活虚拟环境
conda activate day07
# 安装依赖库
pip install transformers==4.57.3 accelerate==1.12.0 -i https://pypi.mirrors.ustc.edu.cn/simple
# 重装Pytorch(Windows)
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
```

### 2、模型下载

```powershell
# 获取下载脚本
https://aliendao.cn/model_download2.py
# 下载模型
python model_download2.py --repo_id Qwen/Qwen3-VL-2B-Instruct
```

### 3、推理测试

```powershell
python qwen3_vl_demo.py
```

## 二、CogVideoX-2B模型

### 1、环境建立

```powershell
# 创建虚拟环境
conda create -n day071 python=3.12 -y
# 激活虚拟环境
conda activate day071
# 安装依赖库
pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple
# 重装Pytorch(Windows)
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
```

### 2、模型下载

```powershell
python model_download2.py --repo_id zai-org/CogVideoX-2b
```

### 3、生成测试

```powershell
# 生成测试
python cog_videox.py
```

