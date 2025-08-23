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
        shortMemory = df.tail(shortMemoryLen).to_dict(orient="records")
    
    # 还差一个树状检索结构，用来检索相似的历史文本内容
    embeddingsTree = None
    
    return shortMemory, embeddingsTree

def getResponse(messages: List[Dict] ,model: str = "qwen-plus", enable_thinking: bool = False) -> str:
    client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key = API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model = model,
        messages = messages,
        extra_body={"enable_thinking": enable_thinking}
    )
    return completion.choices[0].message.content #没有输出think内容

def saveMessages(presaveMessages, embeddingsTree):
    df = pd.DataFrame(presaveMessages)
    df.to_csv(f"{saveDir}/messages.csv", mode="a", index=False, header=False, encoding='utf-8')

def getRelativeMessages(message: str):
    #在查询时，使用树状结构检索
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
    print(getResponse(messages=test_messages))