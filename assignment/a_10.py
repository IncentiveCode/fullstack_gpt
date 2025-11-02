import json
import requests
import time
import streamlit as st
import openai as client
from langchain.utilities import DuckDuckGoSearchAPIWrapper, WikipediaAPIWrapper
from fake_useragent import UserAgent
from bs4 import BeautifulSoup


# meta tag 적용
st.set_page_config(
	page_title="Graduation Project",
	page_icon="🎓",
)

st.title("Graduation Project")


# my functions
def duckduckgo_search(inputs):
	ddg = DuckDuckGoSearchAPIWrapper()
	query = inputs["query"]
	return ddg.run(query)

def wikipedia_search(inputs):
	wiki = WikipediaAPIWrapper()
	query = inputs["query"]
	return wiki.run(query)

def webpage_scrap(inputs):
	url = inputs["url"]
	ua = UserAgent()
	headers = {'User-Agent': ua.random}

	r = requests.get(url, headers=headers)
	soup = BeautifulSoup(r.text, 'html.parser')
	paragraphs = soup.select('p')
	return "\n\n".join(p.get_text() for p in paragraphs)

def text_save(inputs):
	text = inputs["text"]
	file = text.encode('utf-8')

	st.download_button(
		key=time.time(),
		label=f"{query} summary",
		data=file,
		file_name=f"{query}.txt",
		mime="text/txt",
	)


# my function map
functions_map = {
	"duckduckgo_search": duckduckgo_search,
	"wikipedia_search": wikipedia_search,
	"webpage_scrap": webpage_scrap,
	"text_save": text_save,
}

functions = [
	{
		"type": "function",
		"function": {
			"name": "duckduckgo_search",
			"description": "주어진 질문을 duckduckgo 에서 검색 후, 결과를 전달합니다.",
			"parameters": {
				"type": "object",
				"properties": {
					"query": {
						"type": "string",
						"description": "검색하고자 하는 질문",
					}
				},
				"required": ["query"],
			},
		},
	},
	{
		"type": "function",
		"function": {
			"name": "wikipedia_search",
			"description": "주어진 질문을 wikipedia 에서 검색 후, 결과를 전달합니다.",
			"parameters": {
				"type": "object",
				"properties": {
					"query": {
						"type": "string",
						"description": "검색하고자 하는 질문",
					}
				},
				"required": ["query"],
			},
		},
	},
	{
		"type": "function",
		"function": {
			"name": "webpage_scrap",
			"description": "검색 결과 웹페이지를 beautiful soup 을 사용해서 스크랩합니다.",
			"parameters": {
				"type": "object",
				"properties": {
					"url": {
						"type": "string",
						"description": "검색 결과로 찾은 웹페이지의 URL",
					}
				},
				"required": ["url"],
			},
		},
	},
	{
		"type": "function",
		"function": {
			"name": "text_save",
			"description": "스크랩한 결과를 파일로 저장합니다.",
			"parameters": {
				"type": "object",
				"properties": {
					"text": {
						"type": "string",
						"description": "스크랩 한 검색결과 내용",
					}
				},
				"required": ["text"],
			},
		},
	},
]


# session
if "messages" not in st.session_state:
	st.session_state["messages"] = []



# run and message functions
def get_run(run_id, thread_id):
	return client.beta.threads.runs.retrieve(
		run_id=run_id,
		thread_id=thread_id,
	)

def send_message(thread_id, content):
	st.session_state["messages"].append({"message": content, "role": "user"})

	return client.beta.threads.messages.create(
		thread_id=thread_id, role="user", content=content
	)

def get_messages(thread_id):
	messages = client.beta.threads.messages.list(thread_id=thread_id)
	messages = list(messages)
	messages.reverse()
	for message in messages:
		# print(f"{message.role}: {message.content[0].text.value}")
		if message.role == "user":
			with st.chat_message("user"):
				st.write(message.content[0].text.value)
		else:
			st.write(message.content[0].text.value)


def get_tool_outputs(run_id, thread_id):
	run = get_run(run_id, thread_id)
	outputs = []

	try:
		for action in run.required_action.submit_tool_outputs.tool_calls:
			action_id = action.id
			function = action.function
			print(f"Calling function: {function.name} with arg ({json.loads(function.arguments)})")
			outputs.append(
				{
					"output": functions_map[function.name](json.loads(function.arguments)),
					"tool_call_id": action_id,
				}
			)
		return outputs
	except Exception as e:
		print(f"get_tool_outputs(run_id: {run_id}, thread_id: {thread_id}) failure. Unexpected: {e}")
		return None

def submit_tool_outputs(run_id, thread_id):
	outputs = get_tool_outputs(run_id, thread_id)
	if not outputs or outputs.count == 0:
		return

	if not outputs[0] or not "output" in outputs[0]:
		return

	try:
		return client.beta.threads.runs.submit_tool_outputs(
			run_id=run_id, thread_id=thread_id, tool_outputs=outputs
		)
	except Exception as e:
		print(f"submit_tool_outputs(run_id: {run_id}, thread_id: {thread_id}) failure. Unexpected: {e}")
		return


# UI part
with st.sidebar:
	key = st.text_input(
		"OpenAI API Key",
	)

if not key:
	st.markdown("""
		Fullstack GPT 마지막 과제입니다.

		사이드바에 OpenAI API 키를 입력하면, OpenAI Assistants 를 사용할 수 있습니다.
	""")

else:
	# assistants
	if "assistant" not in st.session_state:
		assistant = client.beta.assistants.create(
			name="검색을 위한 조수",
			instructions="당신은 주어진 질문에 대해서 Wikipedia 와 DuckDuckGo 에서 검색을 합니다. 유효한 결과를 찾았다면 내용을 스크랩 후 txt 파일로 저장합니다.",
			model="gpt-3.5-turbo",
			tools=functions
		)
		st.session_state["assistant"] = assistant
	else:
		assistant = st.session_state["assistant"]
	# st.write(f"assistant id: {assistant.id}")
	
	query = st.chat_input("검색하고 싶은 키워드를 입력하세요.")
	if query:
		content = f"이 항목에 대해 검색해줘 : {query}"
		# st.write(f"content: {content}")

		if "thread" not in st.session_state:
			thread = client.beta.threads.create()
			st.session_state["thread"] = thread
		else:
			thread = st.session_state["thread"]
		send_message(thread.id, content)
		# st.write(f"thread id: {thread.id}")
	
		if "run" not in st.session_state:
			run = client.beta.threads.runs.create(
				assistant_id=assistant.id,
				thread_id=thread.id
			)
			st.session_state["run"] = run
		else:
			run = st.session_state["run"]
		# st.write(f"run id: {run.id}")

		while True:
			run = get_run(run.id, thread.id)
			if run.status == "queued" or run.status == "in_progress":
				with st.spinner("진행중..."):
					time.sleep(1)
			elif run.status == "requires_action":
				with st.spinner("스크랩 및 파일 저장 준비..."):
					try:
						submit_tool_outputs(run.id, thread.id)
					except Exception as e:
						print(f"requires_action state(run_id: {run.id}, thread_id: {thread.id}) failure. Unexpected: {e}")
						break
			elif run.status == "completed":
				st.success("준비 완료.")
				get_messages(thread.id)
				break
			else:
				break