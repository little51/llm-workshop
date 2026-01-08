# 第18天 综合实训项目：医疗模型训练与应用系统开发

## 一、环境建立

```powershell
# 创建虚拟环境
conda create -n day18 python=3.12 -y
# 激活虚拟环境
conda activate day18
# 安装依赖库
pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple
# 安装Pytorch
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
# 校验PyTorch
python -c "import torch; print(torch.cuda.is_available())"
```

## 二、下载模型

```powershell
python model_download2.py --repo_id Qwen/Qwen3-1.7B
```

## 三、下载数据集

```powershell
# 下载
python model_download2.py --repo_type dataset --repo_id michaelwzhu/ShenNong_TCM_Dataset
# 目录改名（避免目录名与datasets库冲突）
rename datasets my_datasets
```

## 四、测试原始模型

```powershell
# 启动API Server
python api_server.py --model-name models/Qwen/Qwen3-1.7B
# 开启WebUI测试（另开窗口）
conda activate day18
python med_chat.py
```

## 五、模型训练

```shell
# 激活虚拟环境
conda activate day18
# 安装matplotlib库
pip install matplotlib -i https://pypi.mirrors.ustc.edu.cn/simple
# 训练过程
python med_train_trl.py
# 数据集缓存在C:\Users\<用户名>/.cache/huggingface/datasets/ 
```

## 六、测试训练结果

```powershell
# 激活虚拟环境
conda activate day18
python api_server.py --model-name 模型训练检查点文件夹
# 开启WebUI测试（另开窗口）
conda activate day18
python med_chat.py
```

