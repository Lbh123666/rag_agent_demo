# ========== 1. 导入工具包 ==========
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from config import DEEPSEEK_API_KEY
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import os
import json
# 新增：PDF与纯文本通用加载器
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# ========== 2. 复用本地嵌入模型 ==========
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

# ========== 3. 基础配置 ==========
BASE_URL = "https://api.deepseek.com/v1"
LLM_MODEL = "deepseek-chat"
DB_PATH = "./chroma_db_agent"
# 新增：知识库文档路径，支持 .txt / .pdf
DOC_PATH = "./test_doc.pdf"
# 新增：对话记忆持久化文件路径
MEMORY_FILE = "./chat_history_agent.json"

# ========== 4. 通用文档加载 + 文本切分 ==========
def load_and_split(file_path):
    # 根据后缀自动选择加载器
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        print(f"✅ PDF加载完成，共 {len(docs)} 页")
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        print(f"✅ 文本加载完成，共 {len(docs)} 个文档")
    else:
        raise ValueError("❌ 仅支持 .txt 和 .pdf 格式")

    # 文本切分，参数和原有逻辑一致
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=20
    )
    split_docs = splitter.split_documents(docs)
    # 提取纯文本列表，兼容原有逻辑
    return [doc.page_content for doc in split_docs]

# ========== 5. 构建向量库 ==========
if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
    print("✅ 检测到本地向量库，直接加载...")
    text_chunks = load_and_split(DOC_PATH)
    embedding = LocalEmbedding(dim=256)
    embedding.fit(text_chunks)
    vector_db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding
    )
else:
    print("🔄 本地没有向量库，正在新建...")
    text_chunks = load_and_split(DOC_PATH)
    embedding = LocalEmbedding(dim=256)
    embedding.fit(text_chunks)
    vector_db = Chroma.from_texts(
        texts=text_chunks,
        embedding=embedding,
        persist_directory=DB_PATH
    )
    print("✅ 向量库构建完成")

# 调大粗召回k值，适配简易嵌入模型
retriever = vector_db.as_retriever(search_kwargs={"k": 8})

# ========== 6. 定义知识库检索工具 ==========
@tool
def search_knowledge_base(query: str) -> str:
    """
    知识库检索工具，需要查询新能源汽车品牌知识文档时使用。
    参数query：要检索的关键词或问题
    """
    res_docs = retriever.invoke(query)
    combine_text = "\n".join([d.page_content for d in res_docs])
    return combine_text

tools = [search_knowledge_base]
tool_map = {t.name: t for t in tools}

# ========== 7. 初始化大模型，绑定工具 ==========
llm = ChatOpenAI(
    model=LLM_MODEL,
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base=BASE_URL,
    temperature=0
)
llm_with_tools = llm.bind_tools(tools)

# ========== ✅ 核心：对话记忆 ==========
messages = [
    {
        "role": "system",
        "content": "你是一个新能源汽车知识库问答智能体。常识问题可以直接回答，需要本地文档信息时调用search_knowledge_base工具。请结合历史对话理解用户问题。"
    }
]

# ========== 新增：记忆持久化读写函数 ==========
def save_chat_history():
    """
    把当前对话历史保存到本地JSON文件
    仅保存用户和AI的最终对话，工具调用中间过程不持久化，避免复杂ID匹配
    """
    history = []
    for msg in messages:
        # 跳过系统提示词
        if isinstance(msg, dict) and msg.get("role") == "system":
            continue
        # 用户消息
        if isinstance(msg, HumanMessage):
            history.append({"role": "user", "content": msg.content})
        # AI回复消息
        elif isinstance(msg, AIMessage) and not msg.tool_calls:
            history.append({"role": "assistant", "content": msg.content})
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def load_chat_history():
    """
    程序启动时，从本地文件加载历史对话到记忆中
    """
    if not os.path.exists(MEMORY_FILE):
        print("🔄 未找到历史对话记忆，开启全新对话")
        return
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        print("✅ 已加载本地历史对话记忆")
    except Exception as e:
        print(f"⚠️ 记忆文件加载失败，开启新对话：{e}")

# ========== 8. 主循环：ReAct 手动实现 ==========
if __name__ == "__main__":
    # 程序启动先加载历史记忆
    load_chat_history()
    print("\n===== 带记忆的RAG Agent（支持PDF + 持久化记忆） =====")
    print("输入问题对话，输入 exit 退出\n")
    max_loop = 3
    while True:
        user_input = input("你：")
        if user_input.strip().lower() == "exit":
            # 退出前保存一次记忆
            save_chat_history()
            print("👋 程序退出，记忆已保存")
            break
        if not user_input.strip():
            continue

        # 1. 把用户输入加入记忆
        messages.append(HumanMessage(content=user_input))

        # 2. Agent 思考循环
        current_loop = 0
        final_answer = ""
        while current_loop < max_loop:
            response = llm_with_tools.invoke(messages)
            # 情况A：模型决定调用工具
            if response.tool_calls:
                print(f"[Agent 调用工具：{[t['name'] for t in response.tool_calls]}]")
                messages.append(response)
                # 逐个执行工具
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    if tool_name in tool_map:
                        tool_result = tool_map[tool_name].invoke(tool_args)
                    else:
                        tool_result = "未知工具"
                    messages.append(
                        ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
                    )
                current_loop += 1
            # 情况B：模型给出最终答案
            else:
                final_answer = response.content
                print(f"Agent：{final_answer}\n")
                messages.append(response)
                # 每轮对话结束保存记忆到本地
                save_chat_history()
                break