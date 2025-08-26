import os
from langchain_community.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain

os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

def split_docs(documents, chunk_size = 1000, chunk_overlap = 20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

# Text 파일 로드 후 Chunk 로 분할
docs = split_docs(TextLoader("data/AI.txt").load())

# Chunk 화된 Text 데이터를 로컬 벡터 스토어(RAG)에 저장. 임베딩 모듈 지정
embeddings = SentenceTransformerEmbeddings(model_name = "all-MiniLM-L6-v2")
rag = Chroma.from_documents(docs, embeddings)

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
chain = load_qa_chain(llm, chain_type = "stuff")


query = "AI란?"

# 사용자 질문을 RAG에서 먼저 검색 (이때 내부적으로 사용자질문이 벡터화 되어 검색이 수행되고, 결과는 일반 텍스트로 변환되어 반환)
matching_docs = rag.similarity_search(query)

# LLM에 RAG 검색결과와 사용자 질문을 질의하면 RAG 결과가 반영된 응답을 생성한다.
answer = chain.run(input_documents = matching_docs, question = query);
print(answer)