# 安装GraphRAG
!pip install graphrag

# 1. 准备数据
!mkdir -p ./ragtest/input
!curl https://www.gutenberg.org/cache/epub/24022/pg24022.txt > ./ragtest/input/book.txt

# 2. 初始化GraphRAG工作区
!graphrag init --root ./ragtest

# 3. 设置API密钥（需要在.env文件中设置GRAPHRAG_API_KEY）

# 4. 运行索引流程
!graphrag index --root ./ragtest

# 5. 查询示例
# 全局搜索 - 关于整个故事的问题
!graphrag query --root ./ragtest --method global "整篇小说讲了一个什么故事?"

# 本地搜索 - 关于特定角色的问题
!graphrag query --root ./ragtest --method local "Scrooge是谁，他的主要关系人有哪些？"