import os
import getpass

from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores.milvus import Milvus
from langchain_nvidia_aiplay import NVIDIAEmbeddings
from langchain_nvidia_aiplay import ChatNVIDIA
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
EMBEDDING_MODEL = "nvolveqa_40k"
CHAT_MODEL = "llama2_13b"
HOST = "127.0.0.1"
PORT = "19530"
COLLECTION_NAME = "test"

if os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
    print("Valid NVIDIA_API_KEY already in environment. Delete to reset")
else:
    nvapi_key = getpass.getpass("NVAPI Key (starts with nvapi-): ")
    assert nvapi_key.startswith("nvapi-"), f"{nvapi_key[:5]}... is not a valid key"
    os.environ["NVIDIA_API_KEY"] = nvapi_key

# Read from Milvus Vector Store
embeddings = NVIDIAEmbeddings(model=EMBEDDING_MODEL)
vectorstore = Milvus(connection_args={"host": HOST, "port": PORT},
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever()

# RAG prompt
template = """<s>[INST] <<SYS>>
Use the following context to answer the user's question. If you don't know the answer,
just say that you don't know, don't try to make up an answer.
<</SYS>>
<s>[INST] Context: {context} Question: {question} Only return the helpful
 answer below and nothing else. Helpful answer:[/INST]"
"""
prompt = ChatPromptTemplate.from_template(template)

# RAG
model = ChatNVIDIA(model=CHAT_MODEL)
chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
)


# Add typing for input
class Question(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question)


def _ingest(url: str) -> dict:
    loader = PyPDFLoader(url)
    data = loader.load()

    # Split docs
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    docs = text_splitter.split_documents(data)

    # Insert the documents in Milvus Vector Store
    _ = Milvus.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection_args={"host": HOST, "port": PORT},
    )
    return {}


ingest = RunnableLambda(_ingest)
