import warnings
warnings.filterwarnings("ignore")

import sys
import os

from langchain_community.vectorstores import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA

# เปลี่ยนเป็น import จาก langchain_community
from langchain_community.document_loaders import PyMuPDFLoader

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
    # ==============================
    # 1) โหลดไฟล์ PDF
    # ==============================
    pdf_file_path = "pdf01.pdf"
    loader = PyMuPDFLoader(pdf_file_path)
    data = loader.load()

    # ==============================
    # 2) สร้าง Text Splits
    # ==============================
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)

    # ==============================
    # 3) เลือกโมเดล Embeddings
    # ==============================
    # เปลี่ยนจากการใช้ HuggingFaceEmbeddings เป็น OllamaEmbeddings
    # embedding_model_name = "sentence-transformers/LaBSE"  # LaBSE มันเป็น model ที่รองรับภาษาไทยไง
    # thai_embedding = HuggingFaceEmbeddings(
    #     model_name=embedding_model_name,
    #     model_kwargs={"device": "cpu"}  # เปลี่ยนเป็น "cuda" หากต้องการใช้ GPU
    # )
    
    # ใช้ OllamaEmbeddings แทน
    thai_embedding = OllamaEmbeddings(model="llama3.2:1b")

    # ==============================
    # 4) สร้าง VectorStore
    # ==============================
    with SuppressStdout():
        vectorstore = Chroma.from_documents(
            documents=all_splits,
            embedding=thai_embedding
        )

    # ==============================
    # 5) สร้าง loop สำหรับถาม-ตอบ
    # ==============================
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

        # สร้าง LLM (Ollama)
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

        # ยิงคำถามและรับคำตอบ
        result = qa_chain.invoke({"query": query})
