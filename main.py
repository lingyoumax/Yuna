from datetime import datetime

from tools import init, getLLMResponse, saveMessages, getRelativeMessages, getEmbeddingResponse
from settings import shortMemoryLen, LLMModel, EmbeddingModel

if __name__ =="__main__":
    presavedMessages=[]
    shortMemory, embeddingsGraph = init()
    while True:
        userMessage = input("请输入：")
        if userMessage == "exit":
            break
        shortMemory.append({"role": "user", "content": userMessage})
        presavedMessages.append({"role": "user", "content": userMessage, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        if len(shortMemory) > shortMemoryLen:
            shortMemory = shortMemory[-shortMemoryLen:]
        relativeMessages = getRelativeMessages(userMessage)
        userEmbedding = getEmbeddingResponse(userMessage, EmbeddingModel) 
        response = getLLMResponse(relativeMessages + shortMemory, model = LLMModel)
        print(response) #记得增加流式输出
        shortMemory.append({"role": "assistant", "content": response})
        presavedMessages.append({"role": "assistant", "content": response, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

saveMessages(presavedMessages, embeddingsGraph)