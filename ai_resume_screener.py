# streamlit run ai_resume_screener.py 

import os
from openai import AzureOpenAI

client =  AzureOpenAI(
  azure_endpoint = "https://az-openai-poc-rs.openai.azure.com/", 
  api_key="84061b70ead648eda1aef8286f76f572", 
  api_version="2023-09-15-preview"
)

#----------------------------- UI Starts-----------------------------------------

import streamlit as st

with st.sidebar:
  """ st.image("ncr_atleos.PNG",output_format='auto') """
  st.markdown("""<span style='color: black;font-size: 25px;'>Welcome to - AI Based Resume Screening Chatbot </span>""", unsafe_allow_html=True)
  st.caption("A chatbot powered by Azure OpenAI")

st.header("AI Based Resume Screening Chatbot")
uploaded_files = st.file_uploader(
    "Choose a PDF file",
    accept_multiple_files=True,
    type="pdf"
)
print("No of Files are: ", len(uploaded_files))


# Check if any files are uploaded
if uploaded_files:
    st.write("Files successfully uploaded!!")
    import pickle
    with open("uploaded_files.pkl", 'wb') as f:
        pickle.dump(uploaded_files, f)

jd = st.text_area("Please enter Job Descriptions:")
is_clicked = st.button("Click to Generate Screening Report", type='primary')
    
#----------------------------- UI Ends-----------------------------------------

import PyPDF2

from PyPDF2 import PdfReader

# get the pdf content in readable form
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

import pickle
with open("uploaded_files.pkl", 'rb') as f:
    uploaded_files = pickle.load(f)

raw_text = get_pdf_text(uploaded_files)


def chat_with_gpt(prompt, discussions_his):
    response = client.chat.completions.create(
        model="gpt-35-turbo-16k",

        messages= discussions_his,

        temperature=0.7,
        max_tokens=800, 
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None    
    )
    return response.choices[0].message.content.strip()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


def add_to_chat_history(role, content):
    st.session_state.chat_history.append({"role": role, "content": content})

# Function to display chat history
def display_chat_history():
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

response_container = st.container(
    border=True,
    height=450
)

discussions=[{"role": "system", 
              "content": "You are a helpful assistant in a Recruiter role, Your task is to screen these resumes " + raw_text + " and provide relevant result according to the prompts"
            }]

res_screenresult_ques = "Screen all top candidates who closely match the job description Job Description. For each candidate, display their name, years of experience, education, highest degree, AI rating (on a scale of 1 to 10) and a brief resume summary ONLY in tabular format. Job Description is - " + jd
#print(res_screenresult_ques)
discussions.append({"role": "user", "content":res_screenresult_ques})
res_screenresult = chat_with_gpt(res_screenresult_ques, discussions)

if is_clicked:
    st.write("Resume Screening Report is generated")
    with response_container:
        add_to_chat_history("assistant", "Here's the Resume Screening Report:")
        add_to_chat_history("user", res_screenresult)
        display_chat_history()

discussions2=[{"role": "system", 
              "content": "You are a helpful assistant in Recruiter role who need to analyse these resumes data " + raw_text + " and provide relevant answers according to the user prompts"
              }]

prompt2 = st.chat_input("Ask anything on CV screening...")

if prompt2:
    add_to_chat_history("user", prompt2)
    discussions2.append({"role": "user", "content": prompt2})
    response2 = chat_with_gpt(prompt2, discussions2)
    add_to_chat_history("assistant", response2)
    discussions2.append({"role": "assistant", "content": response2})

    with response_container:
        display_chat_history()


