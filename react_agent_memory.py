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


# ========== 2. 复用你写的本地嵌入模型 ==========
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


# ========== 4. 文档加载切分 ==========
def load_and_split():
    with open("chat_log.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=20
    )
    return splitter.split_text(raw_text)


# ========== 5. 构建向量库 ==========
if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
    text_chunks = load_and_split()
    embedding = LocalEmbedding(dim=256)
    embedding.fit(text_chunks)
    vector_db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding
    )
else:
    text_chunks = load_and_split()
    embedding = LocalEmbedding(dim=256)
    embedding.fit(text_chunks)
    vector_db = Chroma.from_texts(
        texts=text_chunks,
        embedding=embedding,
        persist_directory=DB_PATH
    )
retriever = vector_db.as_retriever(search_kwargs={"k": 2})


# ========== 6. 定义知识库检索工具 ==========
@tool
def search_knowledge_base(query: str) -> str:
    """
    知识库检索工具，需要查询本地聊天记录文档时使用。
    参数query：要检索的关键词或问题
    """
    res_docs = retriever.invoke(query)
    combine_text = "\n".join([d.page_content for d in res_docs])
    return combine_text


tools = [search_knowledge_base]
# 建立工具名 -> 工具对象的映射，方便调用
tool_map = {t.name: t for t in tools}

# ========== 7. 初始化大模型，绑定工具 ==========
llm = ChatOpenAI(
    model=LLM_MODEL,
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base=BASE_URL,
    temperature=0
)
# 把工具绑定给大模型，让它知道可以调用哪些工具
llm_with_tools = llm.bind_tools(tools)

# ========== ✅ 核心：对话记忆（用列表手动维护，简单直观） ==========
# messages 就是完整的对话记忆，包含：系统提示、历史对话、工具调用记录
messages = [
    {
        "role": "system",
        "content": "你是一个知识库问答智能体。常识问题可以直接回答，需要本地文档信息时调用search_knowledge_base工具。请结合历史对话理解用户问题。"
    }
]

# ========== 8. 主循环：ReAct 手动实现 ==========
if __name__ == "__main__":
    print("===== 带记忆的RAG Agent =====")
    print("输入问题对话，输入 exit 退出\n")

    max_loop = 3  # 最多循环调用3次工具，防止死循环

    while True:
        user_input = input("你：")
        if user_input.strip().lower() == "exit":
            print("程序退出")
            break
        if not user_input.strip():
            continue

        # 1. 把用户输入加入记忆
        messages.append({"role": "user", "content": user_input})

        # 2. Agent 思考循环（ReAct）
        current_loop = 0
        while current_loop < max_loop:
            # 调用大模型，让它判断要不要调用工具
            response = llm_with_tools.invoke(messages)

            # 情况A：模型决定调用工具
            if response.tool_calls:
                print(f"[Agent 调用工具：{[t['name'] for t in response.tool_calls]}]")

                # 把模型的工具调用请求加入记忆
                messages.append(response)

                # 逐个执行工具
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]

                    if tool_name in tool_map:
                        # 执行工具函数，拿到结果
                        tool_result = tool_map[tool_name].invoke(tool_args)
                    else:
                        tool_result = "未知工具"

                    # 把工具执行结果回传给记忆
                    messages.append(
                        ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
                    )

                current_loop += 1
                # 回到循环开头，再次调用大模型，让它根据工具结果继续思考

            # 情况B：模型不调用工具，直接给出最终答案
            else:
                print(f"Agent：{response.content}\n")
                messages.append(response)
                break