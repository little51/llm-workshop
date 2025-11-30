# 第5天 大模型量化技术应用

## 一、Ollama安装

```powershell
# 下载链接
https://ollama.com/download/OllamaSetup.exe
# 备选链接
https://aliendao.cn/ollama#
# 拉取模型
ollama pull qwen3
# 运行模型
ollama run qwen3
```

## 二、Open WebUI安装

```powershell
# 1、创建虚拟环境
conda create -n day05 python=3.11 -y
# 2、激活虚拟环境
conda activate day05
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

## 三、基于Streamlit的ChatBot

```powershell
# 1、激活虚拟环境
conda activate day05
# 2、安装依赖库
pip install streamlit==1.51.0 -i https://pypi.mirrors.ustc.edu.cn/simple
# 3、运行Chat_bot
streamlit run chatbot_ollama.py
```

