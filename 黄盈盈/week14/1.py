import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser

# 加载环境变量
load_dotenv(find_dotenv())

# 1. 创建语言模型
model = ChatOpenAI(model="gpt-3.5-turbo")

# 2. 创建提示模板（特定主题：天文科普）
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位天文科普专家，专门回答关于宇宙、星系和天文现象的问题。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# 3. 创建链
chain = prompt | model | StrOutputParser()

# 4. 消息历史管理
store = {}

def get_message_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = RedisChatMessageHistory(
            session_id, 
            url="redis://localhost:6379/0"  # 使用Redis存储历史记录
        )
    return store[session_id]

# 5. 包装链以支持历史记录
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_message_history,
    input_messages_key="input",
    history_messages_key="history",
)

# 6. 对话示例
def run_chat_session(session_id, theme="天文科普"):
    config = {"configurable": {"session_id": session_id}}
    
    print(f"\n=== {theme}聊天系统 ===")
    print("输入 '退出' 结束对话\n")
    
    while True:
        user_input = input("你: ")
        if user_input.lower() == "退出":
            break
            
        response = chain_with_history.invoke(
            {"input": user_input},
            config=config
        )
        
        print(f"\n助手: {response}\n")

# 运行聊天系统
if __name__ == "__main__":
    run_chat_session("astro_chat_1")