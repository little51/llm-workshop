# 第12天 语音模型应用

## 一、安装F5-TTS

```powershell
# 创建虚拟环境
conda create -n day12 python=3.12 -y
# 激活虚拟环境
conda activate day12
# 安装依赖库
pip install f5-tts==1.1.10 -i https://pypi.mirrors.ustc.edu.cn/simple
# 重装Pytorch(Windows)
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
```

## 二、下载模型

```powershell
# 1、获取模型下载脚本
https://aliendao.cn/model_download2.py
# 2、下载声码器模型
python model_download2.py --repo_id charactr/vocos-mel-24khz
# 3、下载频谱生成模型
python model_download2.py --repo_id SWivid/F5-TTS_Emilia-ZH-EN
# 4、下载语音识别模型
python model_download2.py --repo_id openai/whisper-large-v3-turbo
```

## 三、运行F5-TTS测试程序

```powershell
python f5tts_demo.py
```