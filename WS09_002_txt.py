import warnings
warnings.filterwarnings("ignore")

import sys
import os

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA

class SuppressStdout:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


if __name__ == "__main__":
    # 1) โหลดไฟล์ข้อความภาษาไทย 
    text_file_path = "text02.txt"
    loader = TextLoader(text_file_path, encoding="utf-8")
    data = loader.load()

    # 2) สร้าง Text Splits
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)

    # 3) เลือกโมเดล Embeddings ภาษาไทย
    embedding_model_name = "sentence-transformers/LaBSE"

    thai_embedding = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={
            "device": "cpu"  # เลือก "cuda" หากต้องการใช้ GPU
        }
    )

    # 4) สร้าง VectorStore โดยใช้ Embeddings ภาษาไทย
    with SuppressStdout():
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=thai_embedding)

    # 5) สร้าง loop สำหรับถาม-ตอบ
    while True:
        query = input("\nกรุณาป้อนคำถาม : ")
        if query.lower() == "exit":
            break
        if query.strip() == "":
            continue

        # Prompt Template
        template = """Use the following pieces of context to answer the question at the end.
These are your information. If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
{context}
Question: {question}
Helpful Answer:"""
        QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "question"],
            template=template,
        )

        # สร้าง LLM (ใช้ Ollama ตามตัวอย่าง)
        llm = OllamaLLM(
            model="llama3.2:1b",
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
        )

        # สร้าง RetrievalQA
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        )

        # 6) ยิงคำถามและรับคำตอบ
        result = qa_chain.invoke({"query": query})
