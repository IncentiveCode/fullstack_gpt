import streamlit as st

st.set_page_config(
	page_title="Fullstack GPT Home",
	page_icon="🔥",
)

st.title("Welcome to")
st.subheader("my Fullstack GPT Portfolio")
st.markdown(
	"""
		Here are the apps I made:

		- [x] [DocumentGPT](/DocumentGPT)
		- [ ] [PrivateGPT](/PrivateGPT)
		- [ ] [QuizGPT](/QuizGPT)
		- [ ] [SiteGPT](/SiteGPT)
		- [ ] [MeetingGPT](/MeetingGPT)
		- [ ] [InvestorGPT](/InvestorGPT)
	"""
)

select = st.selectbox("select an app", [
	"DocumentGPT", 
	"PrivateGPT",
	"QuizGPT",
	"SiteGPT",
	"MeetingGPT",
	"InvestorGPT",
])

# data flow
from datetime import datetime

st.write(datetime.now);
st.write(select);

input = st.text_input("Hi, what is your name?");
st.write(input);


# chat messages
import time 
 
if "messages" not in st.session_state:
	st.session_state["messages"] = []

def send_message(message, role, save=True):
	with st.chat_message(role):
		st.write(message)
	if save:
		st.session_state["messages"].append({"message": message, "role": role})

for msg in st.session_state["messages"]:
	send_message(
		msg["message"],
		msg["role"],
		save=False
	)

message = st.chat_input("Send a message to the AI")
if message:
	send_message(message, "human")
	time.sleep(2)
	send_message(f"You said: {message}", "ai")

	with st.sidebar:
		st.write(st.session_state)