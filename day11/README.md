# 第11天 大模型数据蒸馏

## 一、环境建立

```powershell
# 创建虚拟环境
conda create -n day11 python=3.12 -y
# 激活虚拟环境
conda activate day11
# 安装依赖库
pip install transformers==4.57.3 datasets==4.4.1 accelerate==1.12.0 torch -i https://pypi.mirrors.ustc.edu.cn/simple
# 重装Pytorch(Windows)
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
```

## 二、数据集及模型下载

```powershell
# 获取下载脚本
https://aliendao.cn/model_download2.py
# 下载数据集
python model_download2.py --repo_type dataset --repo_id AI-MO/NuminaMath-TIR
# 下载模型
python model_download2.py --repo_id Qwen/Qwen2.5-Math-1.5B-Instruct
```

## 三、数据蒸馏过程

```powershell
python data_distill.py
# 结果保存在distilled_data.jsonl
```