import streamlit as st
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.storage import LocalFileStore
from langchain.embeddings import OllamaEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.chat_models import ChatOllama
from langchain.callbacks.base import BaseCallbackHandler

# meta tag 적용
st.set_page_config(
	page_title="PrivateGPT",
	page_icon="👩‍💻",
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
	

# model setting
llm = ChatOllama(
	model="mistral:latest",
	temperature=0.1,
	streaming=True,
	callbacks=[
		ChatCallbackHandler()
	]
)


# file 처리
@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
	# st.write(file)
	file_content = file.read()
	file_path = f"./.cache/private_files/{file.name}"
	# st.write(file_content, file_path)
	with open(file_path, "wb") as f:
		f.write(file_content)

	cache_dir = LocalFileStore(f"./.cache/private_embeddings/{file.name}")
	splitter = CharacterTextSplitter.from_tiktoken_encoder(
			separator="\n",
			chunk_size=600,
			chunk_overlap=100,
	)
	loader = UnstructuredFileLoader(file_path)
	docs = loader.load_and_split(text_splitter=splitter)
	embeddings = OllamaEmbeddings(model="mistral:latest")
	cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
	vectorstore = FAISS.from_documents(docs, cached_embeddings)
	retriever = vectorstore.as_retriever()
	return retriever


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


# document 를 하나의 string 으로 정리
def format_docs(docs):
	return "\n\n".join(document.page_content for document in docs)


# prompt
prompt = ChatPromptTemplate.from_template(
	"""
	Answer the question using ONLY the following context. If you don't know the answer
	jusy say you don't know. DON'T make anything up.

	Context: {context}
	Question: {question}
	"""
)


# UI
st.title("PrivateGPT")
st.markdown(
	"""
		Welcome!

		Use this chatbot to ask questions to an AI about your file!

		Upload your file on the sidebar.
	"""
)

with st.sidebar:
	file = st.file_uploader(
		"Upload a .txt .pdf or .docs file", 
		type=["pdf", "txt", "docx"]
	)

if file:
	retriever = embed_file(file)

	send_message("I'm ready! Ask away!", "ai", save=False)
	paint_history()
	message = st.chat_input("Ask anything about your file...")

	if message:
		send_message(message, "human")
		# docs = retriever.invoke(message)	
		# st.write(docs)
		# docs = "\n\n".join(document.page_content for document in docs)
		# prompt = prompt.format_messages(context=docs, question=message)
		# st.write(prompt)

		chain = {
			"context": retriever | RunnableLambda(format_docs),
			"question": RunnablePassthrough()
		} | prompt | llm

		# response = chain.invoke(message)
		# send_message(response.content, "ai")
		with st.chat_message("ai"):
			response = chain.invoke(message)

else:
	st.session_state["messages"] = []