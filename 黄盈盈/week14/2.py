import os
import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv, find_dotenv

# 加载环境变量
load_dotenv(find_dotenv())

# 1. 创建图书知识库
class BookDatabase:
    def __init__(self):
        self.books = [
            {"title": "三体", "author": "刘慈欣", "desc": "科幻小说，讲述地球文明与三体文明的宇宙博弈"},
            {"title": "平凡的世界", "author": "路遥", "desc": "现实主义小说，描写中国农村青年的奋斗历程"},
            {"title": "活着", "author": "余华", "desc": "讲述一个普通人在中国历史变革中的生存故事"},
            {"title": "百年孤独", "author": "马尔克斯", "desc": "魔幻现实主义代表作，布恩迪亚家族七代人的传奇故事"},
            {"title": "人类简史", "author": "尤瓦尔·赫拉利", "desc": "从认知革命到科学革命的人类发展史"}
        ]
        self.vector_store = self.create_index()
    
    def create_index(self):
        # 创建嵌入模型
        embeddings = OpenAIEmbeddings()
        
        # 准备文档
        documents = []
        for book in self.books:
            content = f"书名:《{book['title']}》\n作者:{book['author']}\n简介:{book['desc']}"
            documents.append(content)
        
        # 创建向量存储
        return FAISS.from_texts(documents, embeddings)
    
    def search_books(self, query, k=3):
        docs = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

# 2. 创建RAG聊天系统
class BookConsultant:
    def __init__(self):
        self.db = BookDatabase()
        self.model = ChatOpenAI(model="gpt-3.5-turbo")
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", (
                "你是一位图书咨询助手，根据以下图书信息回答用户问题：\n"
                "{book_info}\n\n"
                "请根据图书信息提供专业建议，不要编造不存在的信息。"
            )),
            ("human", "{question}"),
        ])
        self.chain = self.prompt_template | self.model | StrOutputParser()
    
    def get_recommendation(self, question):
        # 检索相关图书
        book_info = "\n\n".join(self.db.search_books(question))
        
        # 生成回答
        return self.chain.invoke({
            "book_info": book_info,
            "question": question
        })

# 3. 运行图书咨询系统
def run_book_consultation():
    consultant = BookConsultant()
    
    print("\n=== 图书咨询系统 ===")
    print("输入图书相关问题，系统将根据馆藏图书推荐")
    print("输入 '退出' 结束咨询\n")
    
    while True:
        question = input("你的问题: ")
        if question.lower() == "退出":
            break
            
        response = consultant.get_recommendation(question)
        print(f"\n图书顾问: {response}\n")

if __name__ == "__main__":
    run_book_consultation()