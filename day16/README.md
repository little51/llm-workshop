# 第16天 大模型企业落地应用

## 一、Langflow安装

### 1、环境建立

```powershell
# 创建虚拟环境
conda create -n day16 python=3.13 -y
# 激活虚拟环境
conda activate day16
# 安装依赖库
pip install langflow==1.5.1 fastapi==0.116.1 docling-core[chunking]==2.54.0 -i https://pypi.mirrors.ustc.edu.cn/simple
```

### 2、运行Langflow

```powershell
# 设置大模型基地址
set OPENAI_API_BASE=http://127.0.0.1:11434/v1
# 起动服务
langflow run
# 访问
http://127.0.0.1:7860
```

### 3、模型准备

```powershell
ollama cp qwen3 gpt-4o-mini
```

## 二、Dify安装

### 1、确认Windows版本

```powershell
winver
```

### 2、下载Docker安装程序

```powershell
# 下载地址
https://docs.docker.com/desktop/setup/install/windows-install/
# 下载链接
https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe?utm_source=docker&utm_medium=webreferral&utm_campaign=docs-driven-download-win-amd64&_gl=1*7qcoet*_ga*MTU3ODg2MDQ5NC4xNzMxMzY5ODMx*_ga_XJWPQMJYHQ*czE3NDk2MTA5MzckbzgkZzEkdDE3NDk2MTA5NDEkajU2JGwwJGgw
```

### 3、安装Docker

```powershell
# 1、运行Docker Desktop Installer.exe
# 2、选择以下两个选项
（1）Use WSL 2 instead of Hyper-V (recommended)
（2）Add shortcut to desktop
# 3、安装完成后重启计算机
# 4、运行“Docker Desktop”
```
### 4、更新WSL2

```powershell
# WSL(Windows Subsystem for Linux，在Windows系统中直接运行完整的Linux环境）
wsl --update
```

### 5、配置镜像

```json
{
    "registry-mirrors": [
        "https://docker.registry.cyou",
        "https://docker-cf.registry.cyou",
        "https://dockercf.jsdelivr.fyi",
        "https://docker.jsdelivr.fyi",
        "https://dockertest.jsdelivr.fyi",
        "https://mirror.aliyuncs.com",
        "https://dockerproxy.com",
        "https://mirror.baidubce.com",
        "https://docker.m.daocloud.io",
        "https://docker.nju.edu.cn",
        "https://docker.mirrors.sjtug.sjtu.edu.cn",
        "https://docker.mirrors.ustc.edu.cn",
        "https://mirror.iscas.ac.cn",
        "https://docker.rainbond.cc",
        "https://do.nark.eu.org",
        "https://dc.j8.work",
        "https://dockerproxy.com",
        "https://gst6rzl9.mirror.aliyuncs.com",
        "https://registry.docker-cn.com",
        "http://hub-mirror.c.163.com",
        "http://mirrors.ustc.edu.cn/",
        "https://mirrors.tuna.tsinghua.edu.cn/"
    ],
    "insecure-registries": [
        "registry.docker-cn.com",
        "docker.mirrors.ustc.edu.cn"
    ],
    "debug": true,
    "experimental": false,
    "builder": {
        "gc": {
            "defaultKeepStorage": "20GB",
            "enabled": true
        }
    }
}
```

### 6、获取Dify源码

```powershell
# Clone代码
git clone https://github.com/langgenius/dify
# 切换工作目录
cd dify
# 检出历史版本
git checkout 36b221b
# 修改配置文件
  ## 将.example改名为.env
  ## 修改以下内容
PIP_MIRROR_URL=https://pypi.tuna.tsinghua.edu.cn/simple
# PIP_MIRROR_URL=

```

### 7、运行Dify服务

```powershell
# 切换工作目录
cd docker
# 起动服务
docker compose up -d
```

### 8、模型准备

```powershell
# 拉取推理模型
ollama pull deepseek-r1:1.5b
# 拉取嵌入模型
ollama pull bge-m3
```

