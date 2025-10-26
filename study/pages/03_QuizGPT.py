import json
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers import WikipediaRetriever
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser, output_parser


# meta tag 적용
st.set_page_config(
	page_title="QuizGPT",
	page_icon="⁉️",
)

st.title("QuizGPT")


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


# llm
llm = ChatOpenAI(
	temperature=0.1,
	model="gpt-3.5-turbo-1106",
	streaming=True,
	callbacks=[StreamingStdOutCallbackHandler()]
)

#
#llm = ChatOpenAI(
#	temperature=0.1,
#	model="gpt-3.5-turbo-1106",
#	streaming=True,
#	callbacks=[StreamingStdOutCallbackHandler()]
#).bind(
#	function_call={
#		"name": "create_quiz",
#	},
#	functions=[
#		function,
#	],
#)



# upload file 처리
@st.cache_resource(show_spinner="Loading file...")
def process_file(file):
	file_content = file.read()
	file_path = f"./.cache/quiz_files/{file.name}"
	with open(file_path, "wb") as f:
		f.write(file_content)

	splitter = CharacterTextSplitter.from_tiktoken_encoder(
			separator="\n",
			chunk_size=600,
			chunk_overlap=100,
	)
	loader = UnstructuredFileLoader(file_path)
	docs = loader.load_and_split(text_splitter=splitter)
	return docs

# document 를 하나의 string 으로 정리
def format_docs(docs):
	return "\n\n".join(document.page_content for document in docs)

# output parser
class JsonOutputParser(BaseOutputParser):
	def parse(self, text):
		text = text.replace("```", "").replace("json", "")
		return json.loads(text)

output_parser = JsonOutputParser()


# prompt & chain
question_prompt = ChatPromptTemplate.from_messages([
	(
		"system",
		"""
			당신은 선생님 역할을 수행하는 훌륭한 도우미입니다.
			당신은 뒤에 오는 context 안에서 10개의 문제를 출제합니다.
			모든 문제는 1개의 정답과 3개의 오답을 가집니다.
			정답 선택지는 답 뒤에 (o)를 표기합니다. 

			아래는 예시 문항입니다:

			Q: 바다의 색은 무엇인가요?
			A: 빨간색 | 노란색 | 녹색 | 파란색 (o)

			Q: 대한민국의 수도는 어디인가요?
			A: 대구 | 서울 (o) | 인천 | 부산

			Q: 성남시는 어디에 속한 도시일까요?
			A: 경기도 (o) | 강원도 | 충청북도 | 전라남도

			Q: 일본의 수도는 어디일까요?
			A: 삿포로 | 오사카 | 도쿄 (o) | 오키나와

			자, 이제 당신의 차례입니다.

			context: {context}
		"""
	)
])
question_chain = {"context": format_docs} | question_prompt | llm 

formatting_prompt = ChatPromptTemplate.from_messages([
	(
		"system",
		"""
			당신은 강력한 문서 포맷 알고리즘입니다.
			당신은 퀴즈 정보를 JSON 형태로 제작합니다.

			아래는 예시 입력입니다:

			Q: 바다의 색은 무엇인가요?
			A: 빨간색 | 노란색 | 녹색 | 파란색 (o)

			Q: 대한민국의 수도는 어디인가요?
			A: 대구 | 서울 (o) | 인천 | 부산

			Q: 성남시는 어디에 속한 도시일까요?
			A: 경기도 (o) | 강원도 | 충청북도 | 전라남도

			Q: 일본의 수도는 어디일까요?
			A: 삿포로 | 오사카 | 도쿄 (o) | 오키나와

			아래는 출력 예시입니다:

			```json
			{{
				"questions": [
					{{
						"q": "바다의 색은 무엇인가요?",
						"a": [
							{{
								"answer": "빨간색",
								"correct": false,
							}},
							{{
								"answer": "노란색",
								"correct": false,
							}},
							{{
								"answer": "녹색",
								"correct": false,
							}},
							{{
								"answer": "파란색",
								"correct": true,
							}},
						]
					}},
					{{
						"q": "대한민국의 수도는 어디인가요?",
						"a": [
							{{
								"answer": "대구",
								"correct": false,
							}},
							{{
								"answer": "서울",
								"correct": true,
							}},
							{{
								"answer": "인천",
								"correct": false,
							}},
							{{
								"answer": "부산",
								"correct": false,
							}},
						]
					}},
					{{
						"q": "성남시는 어디에 속한 도시일까요?",
						"a": [
							{{
								"answer": "경기도",
								"correct": true,
							}},
							{{
								"answer": "강원도",
								"correct": false,
							}},
							{{
								"answer": "충청북도",
								"correct": false,
							}},
							{{
								"answer": "전라남도",
								"correct": false,
							}},
						]
					}},
					{{
						"q": "일본의 수도는 어디일까요?",
						"a": [
							{{
								"answer": "삿포로",
								"correct": false,
							}},
							{{
								"answer": "오사카",
								"correct": false,
							}},
							{{
								"answer": "도쿄",
								"correct": true,
							}},
							{{
								"answer": "오키나와",
								"correct": false,
							}},
						]
					}},
				]
			}}
			```

			자, 이제 당신의 차례입니다.

			context: {context}
		"""
	)
])

formatting_chain = formatting_prompt | llm

@st.cache_data(show_spinner="퀴즈를 제작하는 중...")
def make_quiz(_docs, topic):
	chain = {"context": question_chain} | formatting_chain | output_parser
	return chain.invoke(_docs)

@st.cache_data(show_spinner="위키피디아 검색 중...")
def search_wikipedia(term):
	retriever = WikipediaRetriever(top_k_results=5, lang="ko")
	return retriever.get_relevant_documents(term)


# UI part
with st.sidebar:
	docs = None
	topic = None
	choice = st.selectbox(
		"원하는 항목을 선택하세요.",
		(
			"파일",
			"위키피디아 문서",
		),
	)
	if choice == "파일":
		file = st.file_uploader(
			"docx, txt 또는 pdf 파일을 업로드하세요.",
			type=["pdf", "txt", "docx"],
		)
		if file:
			docs = process_file(file)
	else:
		topic = st.text_input("검색하고 싶은 키워드를 입력하세요.")
		if topic:
			docs = search_wikipedia(topic)
	
if not docs:
	st.markdown("""
		QuizGPT 에 방문하신 것을 환영합니다.

		파일 업로드를 하거나 위키피디아 검색을 통해 퀴즈를 만들어 드립니다. 당신은 그것으로 당신의 지식을 테스트 해볼 수 있습니다.

		사이드바에서 원하는 항목을 선택하고 진행하세요.
	""")
else:
	res = make_quiz(docs, topic if topic else file.name)
	
	# prompt = PromptTemplate.from_template("Make a quiz using this context: {docs}")
	# chain = prompt | llm
	# res = chain.invoke({"docs": docs})
	# res = res.additional_kwargs["function_call"]["arguments"]
	# st.write(res)


	with st.form("Quiz Time~"):
		for question in res["questions"]:
			st.write(question["q"])
			value = st.radio(
				f"정답을 찾아보세요.",
				[a["answer"] for a in question["a"]],
				key=f"radio_{question['q']}",
				index=None,
			)

			if {"answer": value, "correct": True} in question["a"]:
				st.success("정답입니다!")
			elif value is not None:
				st.error("틀렸습니다...")
		
		button = st.form_submit_button(label="답안 제출")