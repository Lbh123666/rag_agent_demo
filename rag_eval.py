# ========== 1. 导入依赖，复用原有RAG组件 ==========
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import os

# ========== 2. 复用本地嵌入模型（和rag_demo完全一致） ==========
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

# ========== 3. 配置项 ==========
DB_PATH = "./chroma_db"
DOC_PATH = "./test_doc.pdf"
# 评估Top-K召回效果，和你线上用的粗召回k保持一致
EVAL_K = 8

# ========== 4. 文档加载切分（复用原有逻辑） ==========
def load_and_split(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
    else:
        raise ValueError("仅支持txt、pdf格式")
    splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
    split_docs = splitter.split_documents(docs)
    return [doc.page_content for doc in split_docs]

# ========== 5. 加载向量库 ==========
if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
    text_chunks = load_and_split(DOC_PATH)
    embedding = LocalEmbedding(dim=256)
    embedding.fit(text_chunks)
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embedding)
else:
    text_chunks = load_and_split(DOC_PATH)
    embedding = LocalEmbedding(dim=256)
    embedding.fit(text_chunks)
    vector_db = Chroma.from_texts(texts=text_chunks, embedding=embedding, persist_directory=DB_PATH)

retriever = vector_db.as_retriever(search_kwargs={"k": EVAL_K})

# ========== 6. 测试数据集（基于你的新能源PDF文档定制） ==========
# has_answer=True：文档里有答案，必须召回成功才算对
# has_answer=False：文档里无答案，验证不误召
test_dataset = [
    {
        "question": "比亚迪有哪些核心自研技术？",
        "keywords": ["刀片电池", "DM-i", "CTB", "易四方", "云辇"],
        "has_answer": True
    },
    {
        "question": "刀片电池是什么类型的电池？",
        "keywords": ["磷酸铁锂"],
        "has_answer": True
    },
    {
        "question": "蔚来换电模式有什么优势？",
        "keywords": ["换电", "补能"],
        "has_answer": True
    },
    {
        "question": "800V高压平台的作用是什么？",
        "keywords": ["快充", "充电速度", "800V"],
        "has_answer": True
    },
    {
        "question": "CTB技术是什么意思？",
        "keywords": ["电池车身一体化"],
        "has_answer": True
    },
    {
        "question": "新能源汽车补能方式有哪些？",
        "keywords": ["充电", "换电"],
        "has_answer": True
    },
    {
        "question": "比亚迪的混动技术叫什么？",
        "keywords": ["DM-i"],
        "has_answer": True
    },
    {
        "question": "云辇是哪个公司的技术？",
        "keywords": ["比亚迪"],
        "has_answer": True
    },
    {
        "question": "特斯拉Model 3售价是多少？",
        "keywords": [],
        "has_answer": False
    },
    {
        "question": "理想汽车用的是什么电池？",
        "keywords": [],
        "has_answer": False
    }
]

# ========== 7. 核心评估逻辑 ==========
def evaluate_retrieval():
    total = len(test_dataset)
    success = 0
    bad_cases = []

    print(f"===== RAG 召回效果评估（Top-{EVAL_K}） =====")
    print(f"总测试题数：{total}\n")

    for idx, item in enumerate(test_dataset):
        question = item["question"]
        keywords = item["keywords"]
        has_answer = item["has_answer"]

        # 执行向量检索
        docs = retriever.invoke(question)
        retrieved_text = "\n".join([d.page_content for d in docs])

        # 判断命中结果
        if not has_answer:
            # 文档无答案的题，不召回相关内容算正常通过
            success += 1
            status = "✅ 无答案题，正常未召回"
        else:
            # 有答案的题：只要召回文本包含任意一个关键词就算命中
            hit = any(kw in retrieved_text for kw in keywords)
            if hit:
                success += 1
                status = "✅ 召回成功"
            else:
                status = "❌ 召回失败"
                bad_cases.append({
                    "question": question,
                    "expected_keywords": keywords,
                    "top1_preview": docs[0].page_content[:120] + "..." if docs else "无结果"
                })

        print(f"第{idx+1}题：{question}")
        print(f"  结果：{status}\n")

    # 计算最终指标
    success_rate = round(success / total * 100, 2)

    print("=" * 40)
    print("📊 评估结果汇总")
    print(f"总测试题数：{total}")
    print(f"召回成功数：{success}")
    print(f"召回成功率：{success_rate}%")

    if bad_cases:
        print("\n⚠️ Bad Case 失败清单")
        for case in bad_cases:
            print(f"问题：{case['question']}")
            print(f"预期关键词：{case['expected_keywords']}")
            print(f"召回第一条预览：{case['top1_preview']}\n")

    return success_rate, bad_cases

# ========== 8. 入口：运行评估 ==========
if __name__ == "__main__":
    evaluate_retrieval()