# 第2天 大模型应用环境搭建

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

