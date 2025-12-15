# 第13天 数字人应用

## 一、环境建立

```powershell
# 1、创建虚拟环境
conda create -n day13 python=3.12 -y
# 激活虚拟环境
conda activate day13
# 2、克隆echomimic_v2源码
git clone https://github.com/little51/echomimic_v2
# 3、安装依赖库
pip install -r echomimic_v2\requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple
# 4、降级pydantic
pip install pydantic==2.10.6 -i https://pypi.mirrors.ustc.edu.cn/simple
# 5、安装GPU版PyTorch
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
# 6、验证PyTorch（如果显示True则为正常）
python -c "import torch; print(torch.cuda.is_available())"
```

## 二、下载模型

```powershell
# 1、获取模型下载脚本
https://aliendao.cn/model_download2.py
# 2、下载EchoMimicV2及相关模型
python model_download2.py --repo_id BadToBest/EchoMimicV2
# 3、下载ChatTTS模型
python model_download2.py --repo_id 2Noise/ChatTTS
# 4、下载ffmpeg
https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-7.1.1-essentials_build.zip
```

## 三、运行示例

```powershell
# 设置ffmpeg路径
set FFMPEG_PATH=.\
# 运行Gradio实例
python echomimic_v2\app.py
# 姿态输入（目录地址）
echomimic_v2/assets/halfbody_demo/pose/fight
```
