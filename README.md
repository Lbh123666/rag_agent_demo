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

## 核心功能
1. 基础 RAG 问答
- 文档加载与递归字符切分，支持 chunk 大小与重叠参数调整
- 本地向量库持久化，二次启动无需重复向量化
- 基于相似度召回 Top‑K 文档片段，约束大模型基于原文作答

2. ReAct 智能体
- 自主决策：大模型判断是否需要调用知识库工具
- 多轮记忆：保留完整对话上下文，支持指代性提问
- 循环调用：支持多轮工具调用，信息不足时自动重试检索
- 循环次数限制，防止死循环

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