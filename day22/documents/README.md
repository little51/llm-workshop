# 第22天 检索增强应用

## 一、虚拟环境安装

```shell
# 创建虚拟环境
conda create -n day22 python=3.10 -y
# 激活虚拟环境
conda activate day22
# 安装基础依赖项目
pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple
# 降级Numpy
pip install numpy==1.26.4 -i https://pypi.mirrors.ustc.edu.cn/simple
# 降级其他组件
pip install langchain langchain-community unstructured pdfminer.six==20220319 onnxruntime==1.17.1 -i https://pypi.mirrors.ustc.edu.cn/simple
```

## 二、下载向量化模型文件

```shell
# 模型下载脚本从aliendao.cn首页下载
# 链接为 https://aliendao.cn/model_download.py
# linux下使用wget命令下载，windows下直接在浏览器打开链接下载
wget https://aliendao.cn/model_download2.py
# 从aliendao.cn下载text2vec-base-chinese模型文件
python model_download2.py --repo_id shibing624/text2vec-base-chinese
# 下载后的文件在./models/shibing624/text2vec-base-chinese目录
```

## 三、运行

```shell
# 分词器下载
python -c "import nltk; nltk.download('punkt_tab')"
# 运行测试程序
python rag-demo.py
```

