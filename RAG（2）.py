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


# ========== 3. 基础配置 ==========
#deep seek的api接口地址，兼容openai格式
BASE_URL = "https://api.deepseek.com/v1"
#使用大模型的名称
LLM_MODEL = "deepseek-chat"
#向量库本地保存的文件夹路径，实现持久化（关闭程序再打开数据还在）
DB_PATH = "./chroma_db"  # 向量库存放的文件夹


# ========== 4. 打包：读文件+文本切块 ==========
#打包一个函数：读取本地文件+切成小块
def load_and_split():
    #打开本地chat_log.txt文件，读取全部文本内容→对应文档加载
    with open("chat_log.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    #初始化递归文本切分器
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,#每个文本块最多150个字符
        chunk_overlap=20#相邻两个块之间重叠20个字符，防止上下文断裂
    )
    #执行切分，返回一个列表、每个元素是一小段文本
    return splitter.split_text(raw_text)


# ========== 5. 核心：向量库持久化逻辑 ==========
# 判断：本地有现成向量库就加载，没有就新建
if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
    print("✅ 检测到本地向量库，直接加载...")
    #先拿到切分好的文本块（用来训练embedding词表）
    text_chunks = load_and_split()
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
    text_chunks = load_and_split()
    print(f"文档切块完成，共 {len(text_chunks)} 段")

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

# 从向量库拿到检索器：用来搜相关内容
retriever = vector_db.as_retriever(search_kwargs={"k": 2})# 表示每次召回最相关的 2 段文本

# ========== 6. 初始化大模型和提示词 ==========
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

# ========== 7. 组装RAG问答流水线 ==========
#用LangChain的管道语法，把整个流程串起来
rag_chain = (
        {
            #第一步：用户问题进来→调用检索器拿相关文档→把多段文档用换行拼接成字符串
            "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
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

# ========== 8. 交互式问答循环（今天核心新增） ==========
if __name__ == "__main__":
    print("\n===== RAG知识库问答系统 =====")
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