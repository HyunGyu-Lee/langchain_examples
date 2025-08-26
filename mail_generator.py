import os

import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

llm = ChatOpenAI(temperature=0, model="gpt-4")

st.set_page_config(page_title = "이메일 작성 서비스", page_icon = ":robot:")
st.header("이메일")

def getEmail():
    input_text = st.text_area(label = "메일 입력", label_visibility = "collapsed", placeholder = "당신의 메일은...", key = "input_text")
    return input_text

input_text = getEmail()

query_template = """
    메일을 작성해주세요.
    귀엽고 깜찍한 어조로 작성해주세요.
    아래는 이메일입니다:
    이메일: {email}
"""

prompt = PromptTemplate(input_variables = ['email'], template = query_template)

st.button("*예제를 보여주세요*", type = "secondary", help = "봇이 작성한 메일을 확인해보세요.")
st.markdown("### 봇이 작성한 메일")

if input_text:
    mail_draft = llm.predict(prompt.format(email = input_text))
    st.write(mail_draft)