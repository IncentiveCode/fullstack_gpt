import streamlit as st

st.set_page_config(
	page_title="FullstackGPT Home",
	page_icon="👩‍💻",
)

st.title("FullstackGPT Home")
st.subheader("my Fullstack GPT Portfolio")
st.markdown(
	"""
		Here are the apps I made:

		- [x] [DocumentGPT](/DocumentGPT)
		- [x] [PrivateGPT](/PrivateGPT)
		- [x] [QuizGPT](/QuizGPT)
		- [ ] [SiteGPT](/SiteGPT)
		- [ ] [MeetingGPT](/MeetingGPT)
		- [ ] [InvestorGPT](/InvestorGPT)
	"""
)