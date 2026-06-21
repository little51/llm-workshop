# 第23天 OCR应用

## GLM-OCR

### 1、创建虚拟环境

```shell
conda create -n day23 python=3.12 -y
conda activate day23
```

### 2、安装依赖库

```shell
# 1、安装transformers
# 方法1 #########################
pip install git+https://github.com/huggingface/transformers.git -i https://pypi.mirrors.ustc.edu.cn/simple
################################

# 方法2 #########################
git clone https://gitclone.com/github.com/huggingface/transformers.git --depth=1
# 或
git clone https://github.com/huggingface/transformers.git --depth=1
#
cd transformers
#
pip install -e . -i https://pypi.mirrors.ustc.edu.cn/simple
#
cd ..
################################
# 2、安装PyTorch
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
# 3、安装其他依赖库
pip install Pillow accelerate -i https://pypi.mirrors.ustc.edu.cn/simple
```

### 3、运行程序

```shell
set HF_ENDPOINT=https://hf-mirror.com
python glm_ocr_test.py
```

