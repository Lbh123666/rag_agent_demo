# ========== 1. 导入需要的工具包 ==========
#导入chroma向量数据库，负责存向量、做相似度检索
from langchain_chroma import Chroma
#导入递归文本切分器，负责把长文本切成小块chunk
from langchain_text_splitters import RecursiveCharacterTextSplitter
#导入对话提示词模板，用来组装“上下文+问题”的固定格式
from langchain_core.prompts import ChatPromptTemplate
#透传工具，把用户问题原封不动传到下游
from langchain_core.runnables import RunnablePassthrough
#输出解析器，把大模型返回的对象转成纯文本字符串
from langchain_core.output_parsers import StrOutputParser
#openai格式的大模型调用类，deep seek兼容openai接口
from langchain_openai import ChatOpenAI
#从配置文件导入你的api密钥，避免写代码里
from config import DEEPSEEK_API_KEY
#下面两个是用来实现简易本地词袋向量化的工具
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
# 新增：用来判断本地文件夹是否存在
import os
# ========== PDF与纯文本通用加载器 ==========
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# ========== 2. 本地嵌入模型（纯本地、不用下载，和之前一样） ==========
#自定义一个简易本地嵌入模型，不用联网、下载大模型
class LocalEmbedding:
    def __init__(self, dim=256):#设定向量维度为256维
        self.dim = dim
        #用词袋模型+字符二元组来做简单的文本向量化
        self.vectorizer = CountVectorizer(max_features=dim, analyzer="char_wb", ngram_range=(2, 2))
    # 用文档训练词表，把文字转成数字的规则
    def fit(self, texts):
        self.vectorizer.fit(texts)
    #把一批文本批量转成向量
    def embed_documents(self, texts):
        vectors = self.vectorizer.transform(texts).toarray()
        #维度不够就补0，保证所以向量长度一致
        if vectors.shape[1] < self.dim:
            zeros = np.zeros((vectors.shape[0], self.dim - vectors.shape[1]))
            vectors = np.hstack([vectors, zeros])
        return vectors.tolist()
    def embed_query(self, text):
        # 把单条问题转成向量
        return self.embed_documents([text])[0]

# ========== 3. 简易版 Rerank 重排序精排模型 ==========
# 保留两阶段检索架构，内部用关键词打分实现，零模型下载依赖
# 工业级场景可无缝替换为 bge-reranker 等交叉编码器模型
class Reranker:
    def __init__(self):
        # 无需加载外部模型，纯本地计算，秒启动
        pass
    
    def rerank(self, query: str, docs: list, top_n: int = 2):
        """
        对向量粗召回的候选片段重新打分排序
        query: 用户问题
        docs: 向量召回的Document对象列表
        top_n: 返回最相关的前N个
        """
        if not docs:
            return []
        
        # 把问题拆成字符集合，计算与文档的重合度
        query_chars = set(query)
        scored_list = []
        
        for doc in docs:
            doc_chars = set(doc.page_content)
            # 重合字符占比作为相关性分数
            same_chars = query_chars & doc_chars
            score = len(same_chars) / len(query_chars) if len(query_chars) > 0 else 0
            scored_list.append((score, doc))
        
        # 按分数从高到低排序
        scored_list.sort(key=lambda x: x[0], reverse=True)
        # 返回前 top_n 个最相关文档
        return [doc for score, doc in scored_list[:top_n]]

# ========== 4. 基础配置 ==========
#deep seek的api接口地址，兼容openai格式
BASE_URL = "https://api.deepseek.com/v1"
#使用大模型的名称
LLM_MODEL = "deepseek-chat"
#向量库本地保存的文件夹路径，实现持久化（关闭程序再打开数据还在）
DB_PATH = "./chroma_db"  # 向量库存放的文件夹
# ========== 知识库文档路径，支持 .txt 和 .pdf 格式 ==========
# 切换格式直接改路径即可，函数自动识别
DOC_PATH = "./test_doc.pdf"

# ========== 5. 打包：通用文档加载 + 文本切块 ==========
# 自动识别 txt / pdf，加载后切分，返回纯文本块列表，兼容后续全部原有逻辑
def load_and_split(file_path):
    # 第一步：根据后缀自动选择加载器
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        print(f"✅ PDF加载完成，共 {len(docs)} 页")
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        print(f"✅ 文本加载完成，共 {len(docs)} 个文档")
    else:
        raise ValueError("❌ 目前仅支持 .txt 和 .pdf 格式的文档")

    # 第二步：文本切分（参数和原来保持一致）
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,#每个文本块最多150个字符
        chunk_overlap=20#相邻两个块之间重叠20个字符，防止上下文断裂
    )
    split_docs = splitter.split_documents(docs)
    print(f"✅ 文本切分完成，共 {len(split_docs)} 个块")

    # 第三步：提取纯文本内容，和原有逻辑完全兼容
    text_chunks = [doc.page_content for doc in split_docs]
    return text_chunks

# ========== 6. 核心：向量库持久化逻辑 ==========
# 判断：本地有现成向量库就加载，没有就新建
if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
    print("✅ 检测到本地向量库，直接加载...")
    #先拿到切分好的文本块（用来训练embedding词表）
    text_chunks = load_and_split(DOC_PATH)
    #初始化自定义嵌入模型
    embedding = LocalEmbedding(dim=256)
    embedding.fit(text_chunks)
    # 从本地文件夹直接加载本地已有的向量库
    vector_db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding
    )
else:
    print("🔄 本地没有向量库，正在新建...")
    #加载并切分文档
    text_chunks = load_and_split(DOC_PATH)
    #初始化嵌入模型，用词表训练
    embedding = LocalEmbedding(dim=256)
    embedding.fit(text_chunks)
    # 文本批量向量化，并存入chroma向量库
    vector_db = Chroma.from_texts(
        texts=text_chunks,
        embedding=embedding,
        persist_directory=DB_PATH
    )
    print("✅ 向量库构建完成，已保存到本地")

# 向量粗召回：取4个候选，保证召回率，避免漏答案
retriever = vector_db.as_retriever(search_kwargs={"k": 8})
# 初始化 Rerank 精排器
reranker = Reranker()

# ========== 两阶段检索函数 ==========
def retrieve_and_rerank(query: str):
    """
    第一阶段：向量粗召回4个候选
    第二阶段：Rerank精排，保留最相关2个
    返回拼接好的上下文字符串
    """
    # 第一步：向量粗召回
    docs = retriever.invoke(query)
    # 第二步：重排序精筛
    ranked_docs = reranker.rerank(query, docs, top_n=2)
    # 拼接成字符串，送入提示词
    return "\n\n".join([doc.page_content for doc in ranked_docs])

# ========== 7. 初始化大模型和提示词 ==========
#初始化大模型实例
llm = ChatOpenAI(
    model=LLM_MODEL,
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base=BASE_URL,
    temperature=0#温度=0，回答最严谨、少脑洞，适合知识库回答
)
#自定义RAG提示词模板
rag_prompt = ChatPromptTemplate.from_template("""
请严格根据下面的文档内容回答用户问题。
如果文档中没有相关信息，直接回答“文档中没有相关内容”，禁止编造。
文档内容：
{context}
用户问题：{question}
""")

# ========== 8. 组装RAG问答流水线 ==========
#用LangChain的管道语法，把整个流程串起来
rag_chain = (
        {
            # 替换为两阶段检索：粗召回 + Rerank精排
            "context": lambda query: retrieve_and_rerank(query),
            #第二步：用户问题原封不动透传下去
            "question": RunnablePassthrough()
        }
        #第三步：把context和question填进提示词模板
        | rag_prompt
        #第四步：把填好的提示词发给大模型
        | llm
        #第五步：把模型返回结果转成纯文本
        | StrOutputParser()
)

# ========== 9. 交互式问答循环 ==========
if __name__ == "__main__":
    print("\n===== RAG知识库问答系统（PDF支持 + 两阶段Rerank精排） =====")
    print("输入问题提问，输入 exit 退出程序\n")
   #死循环，持续接受用户输入
    while True:
        user_question = input("请输入问题：")
        # 输入exit就退出程序
        if user_question.strip().lower() == "exit":
            print("👋 程序退出")
            break
        # 空输入跳过，不调用API
        if not user_question.strip():
            continue
        # 异常捕获：出错了不闪退，提示错误后继续运行
        try:
            answer = rag_chain.invoke(user_question)
            print(f"AI回答：{answer}\n")
        except Exception as e:
            print(f"❌ 出错了：{e}")
            print("请重试，或者输入exit退出\n")