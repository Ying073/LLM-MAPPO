import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class DeepSeekClient:
    """DeepSeek API 客户端（论文用 deepseek-reasoner 替代本地 DeepSeek-R1-7B）。"""

    def __init__(self, model: str = "deepseek-reasoner"):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY 未设置：请复制 .env.example 为 .env 并填入密钥")
        self.client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 4000) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return resp.choices[0].message.content
