# 第15天 多种智能体框架应用

## 一、用CrewAI配置一个软件虚拟团队

### 1、环境建立

```powershell
# 创建虚拟环境
conda create -n day151 python=3.13 -y
# 激活虚拟环境
conda activate day151
# 安装依赖库
pip install crewai==0.203.1 -i https://pypi.mirrors.ustc.edu.cn/simple
```

### 2、运行程序

```powershell
python agent_crewai.py
```

## 二、用LangChain开发一个解题Agent

### 1、环境建立

```powershell
# 创建虚拟环境
conda create -n day152 python=3.13 -y
# 激活虚拟环境
conda activate day152
# 安装依赖库
pip install langchain==1.1.2 langchain-community==1.0.0a1 langchain-openai==1.1.0 -i https://pypi.mirrors.ustc.edu.cn/simple
```

### 2、运行程序

```powershell
python agent_langchain.py
```

## 三、用Autogen开发一个解题Agent

### 1、环境建立

```powershell
# 创建虚拟环境
conda create -n day153 python=3.13 -y
# 激活虚拟环境
conda activate day153
# 安装依赖库
pip install autogen-agentchat==0.7.5 autogen-ext[openai]==0.7.5 -i https://pypi.mirrors.ustc.edu.cn/simple
```

### 2、运行程序

```powershell
# 模型准备
ollama cp qwen3 gpt-3.5-turbo
# 运行程序
python agent_autogen.py
```

## 四、用LangGraph开发一个智能客服Agent

### 1、环境建立

```powershell
# 创建虚拟环境
conda create -n day154 python=3.13 -y
# 激活虚拟环境
conda activate day154
# 安装依赖库
pip install langgraph==1.0.4 langchain_openai==1.0.1 -i https://pypi.mirrors.ustc.edu.cn/simple
```

### 2、运行程序

```powershell
python agent_langgraph.py
```

