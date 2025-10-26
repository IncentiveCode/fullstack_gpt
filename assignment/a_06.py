import json
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser, output_parser
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers import WikipediaRetriever


# meta tag 적용
st.set_page_config(
	page_title="QuizGPT Turbo",
	page_icon="⁉️",
)

st.title("QuizGPT Turbo")


# function call
function = {
	"name": "create_quiz",
  "description": "특정 키워드에 대해 10개의 질문을 만들어내는 함수입니다.",
  "parameters": {
    "type": "object",
      "properties": {
        "questions": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "question": {
                "type": "string",
              },
              "answers": {
                "type": "array",
                "items": {
                  "type": "object",
                  "properties": {
                    "answer": {
                      "type": "string",
                    },
                    "correct": {
                      "type": "boolean",
                    },
              	  },
                  "required": ["answer", "correct"],
                },
              },
            },
          "required": ["question", "answers"],
        },
      }
		},
	 	"required": ["questions"],
  },
}

# document 를 하나의 string 으로 정리
def format_docs(docs):
	return "\n\n".join(document.page_content for document in docs)

# output parser
class JsonOutputParser(BaseOutputParser):
	def parse(self, text):
		text = text.replace("```", "").replace("json", "")
		return json.loads(text)

output_parser = JsonOutputParser()



@st.cache_data(show_spinner="퀴즈를 제작하는 중...")
def make_quiz(_docs, topic, difficult):
	prompt = PromptTemplate.from_template("{topic} 에 대해 10개의 문제를 출제합니다. 모든 문제는 1개의 정답과 3개의 오답을 가집니다. 문제의 난이도는 1부터 3까지 있습니다. 현재 난이도 : {difficult}")
	chain = prompt | llm
	return chain.invoke({"topic": topic, "difficult": difficult})

@st.cache_data(show_spinner="위키피디아 검색 중...")
def search_wikipedia(term):
	retriever = WikipediaRetriever(top_k_results=5, lang="ko")
	return retriever.get_relevant_documents(term)


# UI part
with st.sidebar:
	docs = None
	topic = None
	
	key = st.text_input(
		"OpenAI API Key",
	)
	if key:
		llm = ChatOpenAI(
			temperature=0.1,
			openai_api_key=key,
			model="gpt-3.5-turbo-1106",
			streaming=True,
			callbacks=[StreamingStdOutCallbackHandler()]
		).bind(
			function_call={
				"name": "create_quiz",
			},
			functions=[
				function,
			],
		)

		difficult = st.selectbox(
			"난이도를 선택하세요.",
			(
				"1", "2", "3"
			),
		)
		topic = st.text_input("키워드를 입력하세요.")
		if topic:
			docs = search_wikipedia(topic)

		st.text("code")
		code_to_display = """import json
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser, output_parser
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers import WikipediaRetriever


# meta tag 적용
st.set_page_config(
	page_title="QuizGPT Turbo",
	page_icon="⁉️",
)

st.title("QuizGPT Turbo")


# function call
function = {
	"name": "create_quiz",
  "description": "특정 키워드에 대해 10개의 질문을 만들어내는 함수입니다.",
  "parameters": {
    "type": "object",
      "properties": {
        "questions": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "question": {
                "type": "string",
              },
              "answers": {
                "type": "array",
                "items": {
                  "type": "object",
                  "properties": {
                    "answer": {
                      "type": "string",
                    },
                    "correct": {
                      "type": "boolean",
                    },
              	  },
                  "required": ["answer", "correct"],
                },
              },
            },
          "required": ["question", "answers"],
        },
      }
		},
	 	"required": ["questions"],
  },
}

# document 를 하나의 string 으로 정리
def format_docs(docs):
	return "\n\n".join(document.page_content for document in docs)

# output parser
class JsonOutputParser(BaseOutputParser):
	def parse(self, text):
		text = text.replace("```", "").replace("json", "")
		return json.loads(text)

output_parser = JsonOutputParser()



@st.cache_data(show_spinner="퀴즈를 제작하는 중...")
def make_quiz(_docs, topic, difficult):
	prompt = PromptTemplate.from_template("{topic} 에 대해 10개의 문제를 출제합니다. 모든 문제는 1개의 정답과 3개의 오답을 가집니다. 문제의 난이도는 1부터 3까지 있습니다. 현재 난이도 : {difficult}")
	chain = prompt | llm
	return chain.invoke({"topic": topic, "difficult": difficult})

@st.cache_data(show_spinner="위키피디아 검색 중...")
def search_wikipedia(term):
	retriever = WikipediaRetriever(top_k_results=5, lang="ko")
	return retriever.get_relevant_documents(term)


# UI part
with st.sidebar:
	docs = None
	topic = None
	
	key = st.text_input(
		"OpenAI API Key",
	)
	if key:
		llm = ChatOpenAI(
			temperature=0.1,
			openai_api_key=key,
			model="gpt-3.5-turbo-1106",
			streaming=True,
			callbacks=[StreamingStdOutCallbackHandler()]
		).bind(
			function_call={
				"name": "create_quiz",
			},
			functions=[
				function,
			],
		)

		difficult = st.selectbox(
			"난이도를 선택하세요.",
			(
				"1", "2", "3"
			),
		)
		topic = st.text_input("키워드를 입력하세요.")
		if topic:
			docs = search_wikipedia(topic)

		st.text("code")
		code_to_display = \"""
			# CODE
		\"""
		st.code(code_to_display)

if not docs:
	st.markdown(\"""
		QuizGPT 에 방문하신 것을 환영합니다.

		위키피디아 검색을 통해 퀴즈를 만들어 드립니다. 당신은 그것으로 당신의 지식을 테스트 해볼 수 있습니다.

		사이드바에서 원하는 난이도를 선택하고 진행하세요.
	\""")
else:
	res = make_quiz(docs, topic, difficult)
	args = res.additional_kwargs["function_call"]["arguments"]

	correct = 0
	wrong = 0
	with st.form("Quiz Time~"):
		for q in json.loads(args)["questions"]:
			st.write(q["question"])
			value = st.radio(
				f"정답을 찾아보세요.",
				[a["answer"] for a in q["answers"]],
				key=f"radio_{q['question']}",
				index=None,
			)

			if {"answer": value, "correct": True} in q["answers"]:
				st.success("정답입니다!")
				correct += 1 
			elif value is not None:
				st.error("틀렸습니다...")
				wrong += 1
		
		button = st.form_submit_button(label="답안 제출")
	
	if correct == 10:
		st.balloons()
	
		"""
		st.code(code_to_display)

if not docs:
	st.markdown("""
		QuizGPT 에 방문하신 것을 환영합니다.

		위키피디아 검색을 통해 퀴즈를 만들어 드립니다. 당신은 그것으로 당신의 지식을 테스트 해볼 수 있습니다.

		사이드바에서 원하는 난이도를 선택하고 진행하세요.
	""")
else:
	res = make_quiz(docs, topic, difficult)
	args = res.additional_kwargs["function_call"]["arguments"]

	correct = 0
	wrong = 0
	with st.form("Quiz Time~"):
		for q in json.loads(args)["questions"]:
			st.write(q["question"])
			value = st.radio(
				f"정답을 찾아보세요.",
				[a["answer"] for a in q["answers"]],
				key=f"radio_{q['question']}",
				index=None,
			)

			if {"answer": value, "correct": True} in q["answers"]:
				st.success("정답입니다!")
				correct += 1 
			elif value is not None:
				st.error("틀렸습니다...")
				wrong += 1
		
		button = st.form_submit_button(label="답안 제출")
	
	if correct == 10:
		st.balloons()
	