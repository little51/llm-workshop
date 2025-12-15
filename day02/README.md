# 第2天 大模型技术栈与环境配置

## 一、推理卡驱动安装

### 1、下载网址

```powershell
https://www.nvidia.cn/drivers/lookup/
```

### 2、下载选项

依次选择：

GeForce

GeForce RTX 30 Series

GeForce RTX 3060或3060Ti

Windows 10 64-bit

Chinese (Simplified)

### 3、验证

```powershell
nvidia-smi
```

## 二、CUDA安装

### 1、下载网址

```powershell
https://developer.nvidia.com/cuda-toolkit-archive
```

### 2、下载选项

依次选择：

**Operating System：**Linux

**Architecture：**x86_64

**Version：**10

**Installer Type：**exe(local)

### 3、验证

```powershell
nvcc -V
```

## 三、Miniconda安装

### 1、下载网址

```powershell
https://www.anaconda.com/download
```

### 2、下载链接

```powershell
https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
```

### 3、验证

```powershell
conda -V
```

## 四、Git Windows客户端安装

### 1、下载网址

```powershell
https://git-scm.com/book/zh/v2/%E8%B5%B7%E6%AD%A5-%E5%AE%89%E8%A3%85-Git
```

### 2、下载链接

```powershell
https://github.com/git-for-windows/git/releases/download/v2.52.0.windows.1/Git-2.52.0-64-bit.exe
```

### 3、验证

```powershell
git --version
```

## 五、综合练习

```powershell
# 1、创建虚拟环境
conda create -n day02 python=3.12 -y
# 2、激活虚拟环境
conda activate day02
# 3、安装PyTorch
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
# 4、验证是否安装成功
python helloworld.py
```

