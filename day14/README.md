# 第14天 最简智能体应用开发

## 一、任务说明

用10行代码开发一个ADK Agent。

## 二、环境建立

```powershell
# 创建虚拟环境
conda create -n day14 python=3.13 -y
# 激活虚拟环境
conda activate day14
# 安装依赖库
pip install google-adk==1.20.0 litellm==1.80.7 -i https://pypi.mirrors.ustc.edu.cn/simple
```

## 三、模型准备

```powershell
ollama pull qwen3
```

## 四、运行Agent

```powershell
adk web --port 8000
```

