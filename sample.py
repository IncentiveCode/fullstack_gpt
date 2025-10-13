import streamlit as st
# from langchain.prompts import PromptTemplate
from datetime import datetime

now = datetime.today().strftime("%H:%M:%S")
st.title(now)
st.subheader("Welcome to Streamlit!")
st.markdown("""
	#### I love it! 
""")

# st.write("hello")
# a = "hello"
# a

# st.write([1, 2, 3, 4])
# b = [1, 2, 3, 4] 
# b

# st.write({"x": 1})
# c = {"x": 1}
# c

# st.write(PromptTemplate)
# d = PromptTemplate
# d

# p = PromptTemplate.from_template("xxxx")
# st.write(p)
# p

model = st.selectbox("Choose your model", ("GPT-3", "GPT-4"))
if model == "GPT-3":
	st.write("cheap")
else:
	st.write("not cheap")

	name = st.text_input("What is your name?")
	st.write(name)

	value = st.slider("temperature", min_value=0.1, max_value=1.0)
	st.write(value)

with st.sidebar:
	st.title("sidebar title")
	st.text_input("greetings")

tab_one, tab_two, tab_three = st.tabs(["A", "B", "C"])
with tab_one:
	st.write("a")

with tab_two:
	st.write("b")

with tab_three:
	st.write("c")