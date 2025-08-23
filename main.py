from datetime import datetime

from tools import init, getResponse, saveMessages, getRelativeMessages
from settings import shortMemoryLen

if __name__ =="__main__":
    presavedMessages=[]
    shortMemory, embeddingsTree = init()
    while True:
        userMessage = input()
        if userMessage == "exit":
            break
        shortMemory.append({"role": "user", "content": userMessage})
        presavedMessages.append({"role": "user", "content": userMessage, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        if len(shortMemory) > shortMemoryLen:
            del shortMemory[0]
        relativeMessages = getRelativeMessages(userMessage)
        response = getResponse(relativeMessages + shortMemory, enable_thinking=True)
        print(response) #记得增加流式输出
        shortMemory.append({"role": "assistant", "content": response})
        presavedMessages.append({"role": "assistant", "content": response, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

saveMessages(presavedMessages, embeddingsTree)