# 大模型应用与开发实验教程

[电子书下载](https://github.com/little51/llm-workshop/blob/main/%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%BA%94%E7%94%A8%E4%B8%8E%E5%BC%80%E5%8F%91%E5%AE%9E%E9%AA%8C%E6%95%99%E7%A8%8B.pdf)

## 一、课程概览

| 天数   | 主题                 | 内容详情                             |
|--------|----------------------|--------------------------------------|
| [第1天](https://github.com/little51/llm-workshop/tree/main/day01) | 大语言模型导论          | 实训课程概述                         |
|        |                            | 大模型基本概念                      |
|        |                      | 大模型工作原理概要                  |
|        |                      | 大模型应用场景                      |
|        |                      | 大模型应用实践                      |
| [第2天](https://github.com/little51/llm-workshop/tree/main/day02) | 大模型技术栈与环境配置 | 大模型应用架构                      |
|        |                      | 推理卡驱动安装                      |
|        |                      | CUDA安装                           |
|        |                      | Anaconda安装                       |
| [第3天](https://github.com/little51/llm-workshop/tree/main/day03) | 大模型本地部署基础 | 应用环境总结与回顾                  |
|        |                      | Python虚拟环境建立                 |
|        |                      | Xinference依赖库安装           |
|        |                      | Qwen3-0.6B部署                    |
|        |                      | 最简推理程序开发                    |
| [第4天](https://github.com/little51/llm-workshop/tree/main/day04) | 大模型交互式Web应用 | open-webui简介                     |
|        |                      | open-webui部署                     |
|        |                      | open-webui与Qwen3-0.6B组合应用      |
|        |                      | open-webui对话应用场景             |
|        |                      | open-webui知识库应用场景           |
| [第5天](https://github.com/little51/llm-workshop/tree/main/day05) | 大模型量化技术应用     | 大模型量化技术原理介绍              |
|        |                      | Ollama应用架构                     |
|        |                      | Ollama安装                         |
|        |                      | 量化模型部署                       |
|        |                      | open-webui与量化模型组合应用        |
| [第6天](https://github.com/little51/llm-workshop/tree/main/day06) | 大模型企业级部署技术 | 大模型企业级部署特点           |
|        |                      | vLLM介绍               |
|        |                      | vLLM安装  |
|        |                      | vLLM部署模型 |
|        |                      | Chatbot与vLLM组合应用 |
| [第7天](https://github.com/little51/llm-workshop/tree/main/day07) | 多模态模型部署与应用   | 多模态模型简介                      |
|        |                      | 视觉语言模型介绍                    |
|        |                      | Qwen-VL模型部署                    |
|        |                      | 视觉语言模型推理应用开发            |
|        |                      | CogVideoX-2B模型部署与应用开发 |
| [第8天](https://github.com/little51/llm-workshop/tree/main/day08) | 大模型监督微调基础 | 大模型训练原理简介                  |
|        |                      | 大模型训练方法                      |
|        |                      | 最简微调程序与数据解析              |
|        |                      | 训练环境搭建                       |
|        |                      | 完整训练过程实践                   |
| [第9天](https://github.com/little51/llm-workshop/tree/main/day09) | 对话模型专项微调 | 对话模型微调原理                    |
|        |                      | 对话模型微调程序与数据解析          |
|        |                      | 训练环境搭建                       |
|        |                      | 完整训练过程实践                   |
|        |                      | 微调成果应用实践                   |
| [第10天](https://github.com/little51/llm-workshop/tree/main/day10) | 大模型强化学习        | 强化学习基本原理                    |
|        |                      | GRPO算法简介                       |
|        |                      | 强化学习训练过程解读  |
|        |                      | 强化学习训练过程实践 |
|        |                      | 强化学习成果实践                   |
| [第11天](https://github.com/little51/llm-workshop/tree/main/day11) | 大模型数据蒸馏        | 数据蒸馏基本原理                    |
|        |                      | 数据蒸馏过程解析                    |
|        |                      | 数据蒸馏复现原理           |
|        |                      | 数据蒸馏复现过程           |
|        |                      | 数据蒸馏成果实践                   |
| [第12天](https://github.com/little51/llm-workshop/tree/main/day12) | 语音模型与声音克隆    | 语音模型应用场景                    |
|        |                      | 零样本声音克隆原理                  |
|        |                      | 语音模型部署                       |
|        |                      | 零样本声音克隆应用开发              |
|        |                      | 语音模型应用成果实践                |
| [第13天](https://github.com/little51/llm-workshop/tree/main/day13) | 数字人应用         | 数字人应用场景                      |
|        |                      | 半身数字人开发原理                  |
|        |                      | 数字人模型部署                     |
|        |                      | 数字人应用开发                     |
|        |                      | 数字人应用成果实践                  |
| [第14天](https://github.com/little51/llm-workshop/tree/main/day14) | 最简智能体应用开发 | 智能体基础理论                      |
|        |                      | 智能体应用环境搭建                  |
|        |                      | 最简智能体开发实践                  |
|        |                      | 智能体运行原理详解                  |
|        |                      | 智能体与大模型关系分析              |
| [第15天](https://github.com/little51/llm-workshop/tree/main/day15) | 多种智能体框架应用 | 用CrewAI配置一个软件虚拟团队        |
|        |                      | 用LangChain开发一个解题Agent       |
|        |                      | 用Autogen开发一个解题Agent         |
|        |                      | 用LangGraph开发一个智能客服Agent   |
|        |                      | 智能体开发总结                     |
| [第16天](https://github.com/little51/llm-workshop/tree/main/day16) | 大模型企业落地应用      | Dify、Langflow简介                  |
|        |                      | Dify、Langflow安装                  |
|        |                      | Dify、Langflow开发基础应用          |
|        |                      | Dify、Langflow开发智能体应用        |
|        |                      | Dify、Langflow开发工作流应用        |
| [第17天](https://github.com/little51/llm-workshop/tree/main/day17) | 大模型WebUI开发      | 大模型WebUI开发概述                |
|        |                      | SSE原理解析                        |
|        |                      | Gradio框架简介                     |
|        |                      | 前端页面开发（医疗基础应用） |
|        |                      | WebUI与大模型组合应用              |
| [第18天](https://github.com/little51/llm-workshop/tree/main/day18) | 医疗大模型综合项目 | 项目需求分析                     |
|        |                      | 基础大模型部署                     |
|        |                      | 前端页面与模型集成                 |
|        |                      | 医疗训练语料整理                   |
| [第19天](https://github.com/little51/llm-workshop/tree/main/day19) | 医疗大模型综合项目总结 | 医疗领域模型训练                   |
|        |                      | 前端页面开发（医疗进阶应用）         |
|        |                      | 开发文档撰写                       |
|        |                      | 演示准备                           |
| [第20天](https://github.com/little51/llm-workshop/tree/main/day20) | 实训成果展示与评估反馈 | 结业       |

## 二、实验条件

| 类别     | 要求                           | 备注                                    |
| -------- | ------------------------------ | --------------------------------------- |
| 操作系统 | Windows                        | 教程中使用Windows，但也可以使用Linux    |
| 算力设备 | 6G内存以上的显卡或推理卡       | 如GTX1060、RTX2080、RTX4090、P100、T4等 |
| 其他要求 | Windows10 22H2以上（个别章节） | 因为第16章的Dify需要用到Docker环境      |

