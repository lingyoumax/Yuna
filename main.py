from datetime import datetime

from tools import init, getLLMResponse, save, getRelativeMessages, getEmbeddingResponse, updateShortMemory, addEmbedding

def main():
    historyMessages, shortMemory, embeddingsGraph = init()
    while True:
        userMessage = input("请输入：")
        if userMessage == "exit":
            break
        shortMemory.append({"role": "user", "content": userMessage})
        historyMessages.loc[len(historyMessages)] = {"role": "user", "content": userMessage, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        shortMemory = updateShortMemory(shortMemory)    
        userEmbedding = getEmbeddingResponse(userMessage)
        relativeMessages = getRelativeMessages(userEmbedding, historyMessages, embeddingsGraph)
        addEmbedding(embeddingsGraph, userEmbedding)
        response = getLLMResponse(relativeMessages + shortMemory)
        print(response)
        assistantEmbedding = getEmbeddingResponse(response)
        addEmbedding(embeddingsGraph, assistantEmbedding)
        shortMemory.append({"role": "assistant", "content": response})
        historyMessages.loc[len(historyMessages)] = {"role": "assistant", "content": response, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    save(historyMessages, embeddingsGraph)

if __name__ =="__main__":
    main()