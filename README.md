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

## ✨核心功能
1. **基础RAG检索增强**：文档加载、文本分块、向量化存入Chroma本地向量库，检索后交给大模型生成答案
2. **两阶段检索架构**：向量粗召回（k=8）+简易Rerank重排序，过滤无关片段，提升检索质量
3. **多格式文档支持**：兼容PDF、TXT本地知识库文档
4. **ReAct智能体**：Agent自主思考，自行判断什么时候调用知识库检索工具
5. **对话记忆持久化**：多轮对话，对话历史保存到本地JSON，重启程序记忆不会丢失
6. **缓解大模型幻觉**：严格依据检索到的文档回答，无信息直接返回文档无相关内容
7. **RAG效果评估**：内置极简评估脚本，可量化召回成功率，输出Bad Case清单，指导迭代优化
8. **Gradio 可视化 WebUI**：本地网页交互，支持上传 PDF/TXT 文档构建知识库，问答同时展示检索召回的参考片段，便于调试和演示

## 目录结构
```
rag_agent_demo/
├─ rag_demo.py # 基础版 RAG 知识库问答（支持 TXT/PDF，两阶段 Rerank 精排）
├─ react_agent_memory.py # 带对话记忆的 ReAct Agent（自主决策调用工具）
├─ config.py # API 密钥配置（本地使用，不上传 Git）
├─ config_example.py # 密钥配置模板
├─ chat_log_example.txt # 示例知识库文档
├─ requirements.txt # Python 依赖清单
└─ .gitignore # Git 忽略规则
├── rag_webui.py    # Gradio网页前端，支持上传文档、可视化召回参考片段
```

## 运行步骤
1.  安装依赖
    ```bash
    pip install -r requirements.txt
    ```
2.  配置密钥：复制 config_example.py 改名为 config.py，填入你的 DeepSeek API Key
3.  准备知识库：复制 chat_log_example.txt 改名为 chat_log.txt，可替换为自己的文档
4.  运行基础 RAG
    ```bash
    python "rag_demo.py"
    ```
5.  运行带记忆的 Agent
    ```bash
    python react_agent_memory.py
    ```