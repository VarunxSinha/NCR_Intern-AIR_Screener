import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from indexing import retriever

# LangChain setup
# llm = ChatOpenAI(
#     model="gpt-3.5-turbo-0125",
#     api_key=os.getenv('key')
# )
from langchain_openai import AzureChatOpenAI
llm =  AzureChatOpenAI(
    azure_endpoint = "https://az-openai-poc-rs.openai.azure.com/", 
    api_key="84061b70ead648eda1aef8286f76f572", 
    api_version="2023-09-15-preview",
    model = "gpt-35-turbo-16k"
)

# llm = AzureChatOpenAI(
#     azure_endpoint = os.getenv('azure_endpoint'),
#     api_key = os.getenv('api_key'),
#     api_version = os.getenv('api_version'),
#     model = os.getenv('model')
# )

# contextualize_q_system_prompt = (
#     "Given a chat history and the latest user question "
#     "which might reference context in the chat history, "
#     "formulate a standalone question which can be understood "
#     "without the chat history. Do NOT answer the question, "
#     "just reformulate it if needed and otherwise return it as is."
# )

contextualize_q_system_prompt = (
    "As a resume screening assistant, your task is to analyze the uploaded resumes "
    "and give detailed feedback on candidates who match the job description provided: "
    "Focus on relevant skills, experience, and qualifications mentioned in the job description. "
    "Highlight any unique experiences or achievements that align with the job requirements. "
    "Additionally, summarize the key skills for each candidate."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# system_prompt = (
#     "You are an assistant for question-answering tasks. "
#     "Use the following pieces of retrieved context to answer "
#     "the question. If you don't know the answer, say that you "
#     "don't know. Use three sentences maximum and keep the "
#     "answer concise."
#     "\n\n"
#     "{context}"
# )

system_prompt = (
    "Your role is to shortlist the top 2 candidates whose resumes best match the given job description. "
    "For each candidate, provide the following details in a tabular format:\n"
    "- Name\n"
    "- Educational Background\n"
    "- Relevant Skills\n"
    "- Skill Proficiency (beginner, intermediate, expert)\n"
    "- Years of Experience\n"
    "- AI Rating (out of 10)\n"
    "When asked about specific details from the resumes, provide accurate and detailed answers with references from the source. "
    "Generate the response in markdown format."
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Statefully manage chat history
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state["chat_history"]:
        st.session_state["chat_history"][session_id] = ChatMessageHistory()
    return st.session_state["chat_history"][session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

def initialize_state():
    if "generated" not in st.session_state:
        st.session_state["generated"] = ["Hi!, how can I assist you?"]
    if "requests" not in st.session_state:
        st.session_state["requests"] = []
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = {}

def get_response(query):
    answer = conversational_rag_chain.invoke(
        {"input": query},
        config={"configurable": {"session_id": "currentid"}},
    )
    response = answer["answer"]

    # for debugging or checking the context
    # print("-----For Debugging-----")
    # print("Question: ",answer["input"])
    # print(answer["context"])

    st.session_state["chat_history"]["currentid"].messages = st.session_state["chat_history"]["currentid"].messages[-4:]
    return response