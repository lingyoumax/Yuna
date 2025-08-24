from typing import List, Dict
from openai import OpenAI
import os
import csv
import pandas as pd

from config import API_KEY
from settings import saveDir, shortMemoryLen
def init():
    os.makedirs(saveDir, exist_ok=True)

    shortMemory = []
    if not os.path.exists(f"{saveDir}/messages.csv"):
        headers = ["role", "content", "timestamp"]
        with open(f"{saveDir}/messages.csv", mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(headers) 
    else:
        df = pd.read_csv(f"{saveDir}/messages.csv",encoding='utf-8')
        shortMemory = df[['role', 'content']].tail(shortMemoryLen).to_dict(orient="records")
    
    # 还差一个树状检索结构，用来检索相似的历史文本内容
    embeddingsGraph = None
    
    return shortMemory, embeddingsGraph

def getLLMResponse(messages: List[Dict], model: str = "qwen-plus", enable_thinking: bool = False) -> str:
    client = OpenAI(
        api_key = API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model = model,
        messages = messages,
        extra_body={"enable_thinking": enable_thinking}
    )
    return completion.choices[0].message.content

def getEmbeddingResponse(message: str, model: str = "text-embedding-v4", dimensions: int = 1024) -> str:
    client = OpenAI(
        api_key = API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.embeddings.create(
        model = model,
        input = message,
        dimensions = 1024, # 指定向量维度（仅 text-embedding-v3及 text-embedding-v4支持该参数）
        encoding_format="float"
    )
    return completion.data[0].embedding

def saveMessages(presaveMessages, embeddingsGraph):
    df = pd.DataFrame(presaveMessages)
    df.to_csv(f"{saveDir}/messages.csv", mode="a", index=False, header=False, encoding='utf-8')

def getRelativeMessages(message: str):
    #在查询时，使用图结构检索
    #考虑重写查询
    return []

if __name__ == "__main__":
    test_messages = [
        {
            "role": "user",
            "content": "你好，如何使用这个API？",
        },
        {
            "role": "assistant",
            "content": "请提供你的问题，我会尽力解答",
        }
    ]
    print(getLLMResponse(messages=test_messages))