# 第4天 大模型交互式Web应用

## 一、C++编译环境安装

```powershell
# 下载地址
https://visualstudio.microsoft.com/zh-hans/visual-cpp-build-tools/
```

## 二、Open-WebUI安装

```powershell
# 1、创建虚拟环境
conda create -n day04 python=3.11 -y
# 2、激活虚拟环境
conda activate day04
# 3、安装Open WebUI
pip install open-webui==0.6.40 -i https://pypi.mirrors.ustc.edu.cn/simple
# 4、启动Open WebUI服务
# （1）设置环境变量
set HF_ENDPOINT=https://hf-mirror.com
set HF_HUB_OFFLINE=1
# （2）启动服务
open-webui serve
# （3）访问
http://localhost:8080
```

## 三、运行大模型服务

```powershell
# 另开命令行窗口
set XINFERENCE_DISABLE_HEALTH_CHECK=1
set XINFERENCE_HEALTH_CHECK_INTERVAL=300
# 激活虚拟环境
conda activate day03
# 运行
xinference-local --host 127.0.0.1 --port 9997
# 访问
http://127.0.0.1:9997
```

## 四、配置模型参数

```powershell
# Admin 设置 -> 管理员设置 -> 外部连接
# 修改OpenAI连接参数
```

## 五、Open-WebUI应用

```powershell
# 对话
# 上传文件
# 联网搜索
```

