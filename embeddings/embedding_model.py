from langchain.embeddings.base import Embeddings
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

class ZhipuEmbeddings(Embeddings):
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("ZHIPU_API_KEY"),  #  从env读取
            base_url="https://open.bigmodel.cn/api/paas/v4"
        )

    def embed_documents(self, texts):
        response = self.client.embeddings.create(
            model="embedding-2",
            input=texts
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text):
        response = self.client.embeddings.create(
            model="embedding-2",
            input=[text]
        )
        return response.data[0].embedding

    def __call__(self, text):
        return self.embed_query(text)


def get_embedding_model():
    return ZhipuEmbeddings()