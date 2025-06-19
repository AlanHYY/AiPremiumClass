import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)

def test_parameters():
    prompts = ["描述秋天的景色", "用三句话说明AI原理"]
    temps = [0.3, 0.7, 1.0]  # 温度参数
    max_tokens = [50, 100]    # 输出长度
    
    for prompt in prompts:
        for temp in temps:
            for tokens in max_tokens:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temp,
                    max_tokens=tokens
                )
                print(f"\n=== temp={temp}, tokens={tokens} ===")
                print(response.choices[0].message.content)

# 执行测试
test_parameters()



{
  "status": "success",
  "message": "借阅成功",
  "data": {
    "book": "三体",
    "isbn": "97875301",
    "borrow_date": "2025-06-19",
    "return_due": "2025-07-19",
    "remaining_stock": 2
  }
}