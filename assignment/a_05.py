import streamlit as st
import os
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler

# meta tag 적용
st.set_page_config(
	page_title="Assignment #5",
	page_icon="🏁",
)

# 간단한 저장공간 초기화
if "messages" not in st.session_state:
	st.session_state["messages"] = []

# callback class
class ChatCallbackHandler(BaseCallbackHandler):

	def __init__(self):
		self.message = ""

	def on_llm_start(self, *args, **kwargs):
		self.message_box = st.empty()

	def on_llm_new_token(self, token: str, *args, **kwargs):
		self.message += token
		self.message_box.markdown(self.message)

	def on_llm_end(self, *args, **kwargs):
		save_message(self.message, "ai")

# session 에 저장
def save_message(message, role):
	st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
	with st.chat_message(role):
		st.markdown(message)
	if save:
		save_message(message, role)

# session 에 저장된 히스토리 출력
def paint_history():
	for message in st.session_state["messages"]:
		send_message(message["message"], message["role"], save=False)

# file 처리
@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file, key):
	file_content = file.read()
	dir_path = "./.local_cache/"
	os.makedirs(dir_path, exist_ok=True)
	file_path = f"./.local_cache/{file.name}"

	with open(file_path, "wb+") as f:
		f.write(file_content)

	cache_dir = LocalFileStore(file_path)
	splitter = CharacterTextSplitter.from_tiktoken_encoder(
			separator="\n",
			chunk_size=600,
			chunk_overlap=100,
	)
	loader = UnstructuredFileLoader(file_path)
	docs = loader.load_and_split(text_splitter=splitter)
	embeddings = OpenAIEmbeddings(
		openai_api_key=key
	)
	cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
	vectorstore = FAISS.from_documents(docs, cached_embeddings)
	retriever = vectorstore.as_retriever()
	return retriever

# document 를 하나의 string 으로 정리
def format_docs(docs):
	return "\n\n".join(doc.page_content for doc in docs)

# prompt
prompt = ChatPromptTemplate.from_messages([
	("system", 
		"""
		아래의 매우 긴 문서에서 질문에 대한 응답과 관련된 부분을 추출합니다.
		만약 관련있는 문장이 없다면, '' 를 리턴하세요.

		Context: {context}
		"""
	),
	("human", "{question}"),
])


# UI
st.title("FullstackGPT - Assignment #5")
st.markdown(
	"""
		환영합니다!

		이 챗봇에게 파일 내용에 대해서 질문하세요.

		사이드바에서 OpenAI API key 를 입력하고, 파일을 업로드 하면 질문을 할 수 있습니다.
	"""
)

file = st.file_uploader(
	"Upload a .txt .pdf or .docs file", 
	type=["pdf", "txt", "docx"]
)


with st.sidebar:
	key = st.text_input(
		"OpenAI API Key",
	)

	if key:
		# model setting
		llm = ChatOpenAI(
			temperature=0.1,
			openai_api_key=key,
			streaming=True,
			callbacks=[
				ChatCallbackHandler()
			]
		)
	
	st.text("code")
	code_to_display = """import streamlit as st
import os
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler

# meta tag 적용
st.set_page_config(
	page_title="Assignment #5",
	page_icon="🏁",
)

# 간단한 저장공간 초기화
if "messages" not in st.session_state:
	st.session_state["messages"] = []

# callback class
class ChatCallbackHandler(BaseCallbackHandler):

	def __init__(self):
		self.message = ""

	def on_llm_start(self, *args, **kwargs):
		self.message_box = st.empty()

	def on_llm_new_token(self, token: str, *args, **kwargs):
		self.message += token
		self.message_box.markdown(self.message)

	def on_llm_end(self, *args, **kwargs):
		save_message(self.message, "ai")

# session 에 저장
def save_message(message, role):
	st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
	with st.chat_message(role):
		st.markdown(message)
	if save:
		save_message(message, role)

# session 에 저장된 히스토리 출력
def paint_history():
	for message in st.session_state["messages"]:
		send_message(message["message"], message["role"], save=False)

# file 처리
@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file, key):
	file_content = file.read()
	dir_path = "./.local_cache/"
	os.makedirs(dir_path, exist_ok=True)
	file_path = f"./.local_cache/{file.name}"

	with open(file_path, "wb+") as f:
		f.write(file_content)

	cache_dir = LocalFileStore(file_path)
	splitter = CharacterTextSplitter.from_tiktoken_encoder(
			separator="\n",
			chunk_size=600,
			chunk_overlap=100,
	)
	loader = UnstructuredFileLoader(file_path)
	docs = loader.load_and_split(text_splitter=splitter)
	embeddings = OpenAIEmbeddings(
		openai_api_key=key
	)
	cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
	vectorstore = FAISS.from_documents(docs, cached_embeddings)
	retriever = vectorstore.as_retriever()
	return retriever

# document 를 하나의 string 으로 정리
def format_docs(docs):
	return "\n\n".join(doc.page_content for doc in docs)

# prompt
prompt = ChatPromptTemplate.from_messages([
	("system", 
		\"""
		아래의 매우 긴 문서에서 질문에 대한 응답과 관련된 부분을 추출합니다.
		만약 관련있는 문장이 없다면, '' 를 리턴하세요.

		Context: {context}
		\"""
	),
	("human", "{question}"),
])


# UI
st.title("FullstackGPT - Assignment #5")
st.markdown(
	\"""
		환영합니다!

		이 챗봇에게 파일 내용에 대해서 질문하세요.

		사이드바에서 OpenAI API key 를 입력하고, 파일을 업로드 하면 질문을 할 수 있습니다.
	\"""
)

file = st.file_uploader(
	"Upload a .txt .pdf or .docs file", 
	type=["pdf", "txt", "docx"]
)


with st.sidebar:
	key = st.text_input(
		"OpenAI API Key",
	)

	if key:
		# model setting
		llm = ChatOpenAI(
			temperature=0.1,
			openai_api_key=key,
			streaming=True,
			callbacks=[
				ChatCallbackHandler()
			]
		)
	
	st.text("code")
	code_to_display = \"""
		# code
	\"""
	st.code(code_to_display)

if file and key:
	retriever = embed_file(file, key)

	send_message("준비되었습니다! 질문이 있으신가요?", "ai", save=False)
	paint_history()
	message = st.chat_input("파일 내용에 대해서 질문하세요.")

	if message:
		send_message(message, "human")

		chain = {
			"context": retriever | RunnableLambda(format_docs),
			"question": RunnablePassthrough()
		} | prompt | llm

		with st.chat_message("ai"):
			chain.invoke(message)

else:
	st.session_state["messages"] = []
	"""
	st.code(code_to_display)

if file and key:
	retriever = embed_file(file, key)

	send_message("준비되었습니다! 질문이 있으신가요?", "ai", save=False)
	paint_history()
	message = st.chat_input("파일 내용에 대해서 질문하세요.")

	if message:
		send_message(message, "human")

		chain = {
			"context": retriever | RunnableLambda(format_docs),
			"question": RunnablePassthrough()
		} | prompt | llm

		with st.chat_message("ai"):
			chain.invoke(message)

else:
	st.session_state["messages"] = []