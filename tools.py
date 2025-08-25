from typing import List, Dict
from openai import OpenAI
import os
import csv
import faiss
import numpy as np
import pandas as pd

from config import API_KEY
from settings import saveDir, shortMemoryLen, EmbeddingDimension, LLMModel, EmbeddingModel, topK

def init():
    os.makedirs(saveDir, exist_ok=True)

    csv_path = f"{saveDir}/messages.csv"
    if not os.path.exists(csv_path): 
        historyMessages = pd.DataFrame(columns=["role", "content", "timestamp"])
        shortMemory = []
    else:
        historyMessages = pd.read_csv(csv_path, encoding='utf-8')
        shortMemory = historyMessages[['role', 'content']].tail(shortMemoryLen).to_dict(orient="records")

    index_path = os.path.join(saveDir, "faiss_hnsw.index")

    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        if not isinstance(index, (faiss.IndexIDMap2,)):
            index = faiss.IndexIDMap2(index)
        dim = index.d
    else:
        dim = EmbeddingDimension
        M = 32
        efConstruction = 200
        base = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
        base.hnsw.efConstruction = efConstruction
        index = faiss.IndexIDMap2(base)

    next_id = len(historyMessages)

    embeddingsGraph = {
        "index": index,
        "dim": dim,
        "next_id": next_id,
        "paths": {"index": index_path, "csv": csv_path}
    }
    return historyMessages, shortMemory, embeddingsGraph

def updateShortMemory(shortMemory: List[Dict]) -> List[Dict]:
    if len(shortMemory) > shortMemoryLen:
        shortMemory = shortMemory[-shortMemoryLen:]
    return shortMemory
    
def getLLMResponse(messages: List[Dict], enable_thinking: bool = False) -> str:
    client = OpenAI(
        api_key = API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model = LLMModel,
        messages = messages,
        extra_body={"enable_thinking": enable_thinking}
    )
    return completion.choices[0].message.content

def getEmbeddingResponse(message: str) -> str:
    client = OpenAI(
        api_key = API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.embeddings.create(
        model = EmbeddingModel,
        input = message,
        dimensions = EmbeddingDimension, # 指定向量维度（仅 text-embedding-v3及 text-embedding-v4支持该参数）
        encoding_format="float"
    )
    return completion.data[0].embedding

def addEmbedding(embeddingsGraph: Dict, embedding: List[float]):

    index = embeddingsGraph["index"]
    dim = embeddingsGraph["dim"]
    assert len(embedding) == dim
    np_embedding = np.array([embedding], dtype="float32")
    faiss.normalize_L2(np_embedding)

    ids = np.array([embeddingsGraph["next_id"]], dtype="int64")
    index.add_with_ids(np_embedding.astype('float32'), ids)

    embeddingsGraph["next_id"] += 1

def getRelativeMessages(userEmbedding: List[float], historyMessages: pd.DataFrame, embeddingsGraph: Dict):
    """
    根据用户向量在 FAISS 中检索最相近的 topK 条消息，
    返回包含 id/score/role/content/timestamp 的列表。
    约定：FAISS 中的向量 id == historyMessages 的行号（IndexIDMap2 + add_with_ids 保证）。
    """
    index = embeddingsGraph["index"]
    dim = embeddingsGraph["dim"]

    # 基本校验
    if len(userEmbedding) != dim:
        raise ValueError(f"userEmbedding 维度应为 {dim}，实际为 {len(userEmbedding)}")
    if getattr(index, "ntotal", 0) == 0 or len(historyMessages) == 0 or topK <= 0:
        return []

    # 需要排除的 id 区间：最后 shortMemoryLen 条（与 CSV 行号对齐）
    n_rows = len(historyMessages)
    if shortMemoryLen > 0:
        forbid_start = max(0, n_rows - shortMemoryLen)
        forbidden_ids = set(range(forbid_start, n_rows))
    else:
        forbidden_ids = set()

    q = np.asarray([userEmbedding], dtype="float32")
    faiss.normalize_L2(q)

    extra = min(index.ntotal, len(forbidden_ids)) + 10
    k_query = int(min(index.ntotal, max(topK, 1) + extra))

    D, I = index.search(q, k_query)
    ids = I[0].tolist()
    scores = D[0].tolist()

    results = []
    for _id, score in zip(ids, scores):
        if _id == -1:
            continue
        if _id < 0 or _id >= n_rows:
            continue
        if _id in forbidden_ids:
            continue

        row = historyMessages.iloc[int(_id)]
        results.append({
            "role": row.get("role"),
            "content": row.get("content"),
            "timestamp": row.get("timestamp")
        })
        if len(results) >= topK:
            break

    results = sorted(results, key=lambda x: x["timestamp"])
    results = [{"role": r["role"], "content": r["content"]} for r in results]

    return results

def save(historyMessages: pd.DataFrame, embeddingsGraph: Dict):
    historyMessages.to_csv(f"{saveDir}/messages.csv", mode="w", index=False, header=True, encoding='utf-8')
    faiss.write_index(embeddingsGraph["index"], embeddingsGraph["paths"]["index"])
