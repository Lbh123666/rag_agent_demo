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

## 目录结构
```
rag_agent_demo/
├─ RAG (2).py # 基础版 RAG 知识库问答（固定单轮检索链路）
├─ react_agent_memory.py # 带对话记忆的 ReAct Agent（自主决策调用工具）
├─ config.py # API 密钥配置（本地使用，不上传 Git）
├─ config_example.py # 密钥配置模板
├─ chat_log_example.txt # 示例知识库文档
├─ requirements.txt # Python 依赖清单
└─ .gitignore # Git 忽略规则
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
    python "RAG (2).py"
    ```
5.  运行带记忆的 Agent
    ```bash
    python react_agent_memory.py
    ```