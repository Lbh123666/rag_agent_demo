# ========== 1. 导入依赖 ==========
import gradio as gr
import os
import shutil
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from config import DEEPSEEK_API_KEY

# ========== 2. 复用原有RAG核心类（和rag_demo.py完全一致） ==========
class LocalEmbedding:
    def __init__(self, dim=256):
        self.dim = dim
        self.vectorizer = CountVectorizer(max_features=dim, analyzer="char_wb", ngram_range=(2, 2))
    def fit(self, texts):
        self.vectorizer.fit(texts)
    def embed_documents(self, texts):
        vectors = self.vectorizer.transform(texts).toarray()
        if vectors.shape[1] < self.dim:
            zeros = np.zeros((vectors.shape[0], self.dim - vectors.shape[1]))
            vectors = np.hstack([vectors, zeros])
        return vectors.tolist()
    def embed_query(self, text):
        return self.embed_documents([text])[0]

class Reranker:
    def __init__(self):
        pass
    def rerank(self, query: str, docs: list, top_n: int = 2):
        if not docs:
            return []
        query_chars = set(query)
        scored_list = []
        for doc in docs:
            doc_chars = set(doc.page_content)
            same_chars = query_chars & doc_chars
            score = len(same_chars) / len(query_chars) if len(query_chars) > 0 else 0
            scored_list.append((score, doc))
        scored_list.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_list[:top_n]]

# ========== 3. 基础配置 ==========
BASE_URL = "https://api.deepseek.com/v1"
LLM_MODEL = "deepseek-chat"
# 单独的网页版向量库路径，和控制台版本隔离，互不影响
DB_PATH = "./chroma_db_web"

# 初始化模型实例
embedding = LocalEmbedding(dim=256)
reranker = Reranker()

llm = ChatOpenAI(
    model=LLM_MODEL,
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base=BASE_URL,
    temperature=0
)

rag_prompt = ChatPromptTemplate.from_template("""
请严格根据下面的文档内容回答用户问题。
如果文档中没有相关信息，直接回答“文档中没有相关内容”，禁止编造。
文档内容：
{context}
用户问题：{question}
""")

# ========== 4. 核心业务函数1：上传文档构建向量库 ==========
def build_knowledge_base(upload_file):
    """
    接收网页上传的文件，加载切分并构建向量库
    返回构建状态提示
    """
    if upload_file is None:
        return "⚠️ 请先上传 PDF 或 TXT 文档！"
    
    file_path = upload_file.name
    
    # 加载并切分文档
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
    else:
        return "❌ 仅支持 PDF 和 TXT 格式文档"
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
    split_docs = splitter.split_documents(docs)
    text_chunks = [doc.page_content for doc in split_docs]
    
    # 训练嵌入模型词表
    embedding.fit(text_chunks)
    
    # 如果已有旧向量库，先删除再重建
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
    
    # 构建向量库
    Chroma.from_texts(
        texts=text_chunks,
        embedding=embedding,
        persist_directory=DB_PATH
    )
    
    return f"✅ 知识库构建成功，共切分为 {len(split_docs)} 个文本块"

# ========== 5. 核心业务函数2：问答 + 展示召回片段 ==========
def chat_answer(question):
    """
    接收用户问题，执行两阶段检索，调用大模型生成答案
    返回：AI答案 + 召回参考片段
    """
    if not os.path.exists(DB_PATH):
        return "⚠️ 请先上传文档并构建知识库！", ""
    
    # 加载向量库
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embedding)
    retriever = vector_db.as_retriever(search_kwargs={"k": 8})
    
    # 第一阶段：向量粗召回
    raw_docs = retriever.invoke(question)
    # 第二阶段：Rerank精排
    ranked_docs = reranker.rerank(question, raw_docs, top_n=2)
    context = "\n\n".join([doc.page_content for doc in ranked_docs])
    
    # 调用大模型生成答案
    rag_chain = (
        {"context": lambda q: context, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    answer = rag_chain.invoke(question)
    
    # 格式化召回片段，用于网页展示
    ref_text = ""
    for idx, doc in enumerate(ranked_docs):
        ref_text += f"【参考片段 {idx+1}】\n{doc.page_content}\n" + "-" * 60 + "\n"
    
    return answer, ref_text

# ========== 6. Gradio 网页界面布局 ==========
with gr.Blocks(title="RAG知识库可视化问答") as demo:
    gr.Markdown("# 📚 RAG 知识库可视化问答系统")
    gr.Markdown("上传PDF/TXT文档构建知识库，提问并查看检索召回的参考片段")
    
    with gr.Row():
        # 左侧：文档上传区
        with gr.Column(scale=1):
            file_upload = gr.File(label="上传知识库文档（PDF / TXT）")
            build_btn = gr.Button("构建知识库", variant="primary")
            status_text = gr.Textbox(label="构建状态", interactive=False, lines=2)
        
        # 右侧：问答区
        with gr.Column(scale=2):
            question_input = gr.Textbox(label="请输入你的问题", placeholder="例如：比亚迪有哪些核心自研技术？")
            ask_btn = gr.Button("提问", variant="primary")
            answer_output = gr.Textbox(label="AI 回答", interactive=False, lines=6)
            ref_output = gr.Textbox(label="🔍 向量检索召回的参考片段", interactive=False, lines=10)
    
    # 绑定按钮事件
    build_btn.click(
        fn=build_knowledge_base,
        inputs=[file_upload],
        outputs=[status_text]
    )
    
    ask_btn.click(
        fn=chat_answer,
        inputs=[question_input],
        outputs=[answer_output, ref_output]
    )

# ========== 7. 启动网页服务 ==========
if __name__ == "__main__":
    print("🚀 网页服务启动中，浏览器将自动打开...")
    demo.launch(server_name="0.0.0.0", server_port=7860)