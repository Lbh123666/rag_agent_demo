# RAG + ReAct Agent 知识库Demo
基于 LangChain + Chroma 实现的本地知识库问答智能体，用于大模型应用开发学习与面试项目展示。

## 项目简介
本项目从零实现了基础 RAG 检索增强生成，并在此基础上搭建了带对话记忆的 ReAct 智能体。
智能体可以自主判断是否调用知识库检索工具，支持多轮上下文对话，缓解大模型幻觉问题。

## 技术栈
- 大模型：DeepSeek（兼容 OpenAI 接口协议）
- 应用框架：LangChain
- 向量数据库：Chroma（本地持久化）
- 向量化：自定义简易词袋 Embedding（Demo 演示用）
- 核心范式：RAG 检索增强生成 + ReAct 工具调用 Agent
- 部署支持：Docker 容器化一键部署

## ✨核心功能
1. **基础RAG检索增强**：文档加载、文本分块、向量化存入Chroma本地向量库，检索后交给大模型生成答案
2. **两阶段检索架构**：向量粗召回（k=8）+简易Rerank重排序，过滤无关片段，提升检索质量
3. **多格式文档支持**：兼容PDF、TXT本地知识库文档
4. **ReAct智能体**：Agent自主思考，自行判断什么时候调用知识库检索工具
5. **对话记忆持久化**：多轮对话，对话历史保存到本地JSON，重启程序记忆不会丢失
6. **缓解大模型幻觉**：严格依据检索到的文档回答，无信息直接返回文档无相关内容
7. **RAG效果评估**：内置极简评估脚本，可量化召回成功率，输出Bad Case清单，指导迭代优化
8. **Gradio 可视化 WebUI**：本地网页交互，支持上传 PDF/TXT 文档构建知识库，问答同时展示检索召回的参考片段，便于调试和演示
9. **Docker容器化部署**：支持镜像打包一键启动，解决环境不一致问题，API密钥通过环境变量安全注入

## 目录结构
```
rag_agent_demo/
├─ rag_demo.py # 基础版 RAG 知识库问答（支持 TXT/PDF，两阶段 Rerank 精排）
├─ react_agent_memory.py # 带对话记忆的 ReAct Agent（自主决策调用工具）
├─ rag_webui.py # Gradio 网页前端，支持上传文档、可视化召回参考片段
├─ rag_eval.py # RAG 召回效果评估脚本
├─ config.py # API 密钥配置（自动读取 .env 环境变量）
├─ config_example.py # 密钥配置模板
├─ .env # 环境变量文件（存放 API 密钥，不上传 Git）
├─ requirements.txt # Python 依赖清单
├─ Dockerfile # Docker 镜像构建文件
├─ .dockerignore # Docker 构建忽略规则
├─ chat_log_example.txt # 示例知识库文档
├─ test_doc.pdf # 示例 PDF 知识库
└─ .gitignore # Git 忽略规则
```

## 运行步骤
1.  安装依赖
    ```bash
    pip install -r requirements.txt
    ```
2.  配置密钥：在项目根目录新建 .env 文件，填入你的 DeepSeek API Key
    ```bash
    DEEPSEEK_API_KEY = "你的DeepSeek API密钥"
    ```
3.  准备知识库：可使用自带的 test_doc.pdf，也可替换为自己的 PDF/TXT 文档
4.  运行基础 RAG
    ```bash
    python "rag_demo.py"
    ```
5.  运行带记忆的 Agent
    ```bash
    python react_agent_memory.py
    ```
6.  运行 Gradio 网页可视化
    ```bash
    python rag_webui.py
    ```

Docker 一键部署
1. 构建镜像
在项目根目录执行命令，打包生成镜像：
    ```bash
    docker build -t rag-agent-demo .
    ```
2. 启动容器
通过环境变量传入 API 密钥，映射本地端口：
    ```bash
    docker run -d -p 7860:7860 -e DEEPSEEK_API_KEY="你的DeepSeek API密钥" rag-agent-demo .
    ```
3. 访问服务
打开浏览器访问：
    ```bash
    http://localhost:7860
    ```