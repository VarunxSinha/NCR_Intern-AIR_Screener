import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

ncrlogo = "https://media.licdn.com/dms/image/v2/D4E16AQFZOPB4N0lRcg/profile-displaybackgroundimage-shrink_200_800/profile-displaybackgroundimage-shrink_200_800/0/1693953632895?e=2147483647&v=beta&t=4vdbblg-vp1dPuX5t85RknWcbQh7CKiwxtSuugyXa_o"
st.logo(ncrlogo,link="https://www.ncratleos.com/")

st.set_page_config(page_title="AIR-S")

with st.sidebar:
    # st.title("AIR-S")
    add_vertical_space(2)
    st.markdown(
        """
        Aritificial Intelligence Resume-Screening or AIR-S is an in-house developed software powered by LLM to help the Recruiting division of NCR Atleos in 
        screening the Resume/CV's of the applicants for a specific role.
        The software provides with feature of additional question answers and prompt generation according to the recruiters demand.
        """
    )
    add_vertical_space(2)
    

st.header("AIR-S")

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

    from chat_logic import initialize_state, get_response, conversational_rag_chain

    # Initialize the state variables
    initialize_state()

    # button for generating report
    # Input field for job role
    job_role = st.text_input("Job Role")
    report_container = st.container()
    if st.button("Generate Report!!",type="primary"):
        
        answer = conversational_rag_chain.invoke(
            {"input": f"Shortlist top 2 candidates for the {job_role} based on resumes. Provide name, education and AI rating (out of 10) in a table.Give a 3 line summary for each candidate. Generate the response in markdown format."},
            config={"configurable": {"session_id": "currentid"}},
        )["answer"]
        with report_container:  
            st.markdown(answer)
    # Containers for the app
    response_container = st.container(
        border=True,
        height=450
    )
    input_container = st.container()
    

    with input_container:
        query = st.chat_input("Got a question? Fire away!")
        if query:
            with st.spinner("Typing...."):
                response = get_response(query)
            st.session_state.requests.append(query)
            st.session_state.generated.append(response)

    with response_container:
        if st.session_state["generated"]:
            for i in range(len(st.session_state["generated"])):
                response_container.chat_message("assistant").write(st.session_state["generated"][i])
                if i < len(st.session_state["requests"]):
                    response_container.chat_message("user").write(st.session_state["requests"][i])

else:
    st.write("No files uploaded yet.")
