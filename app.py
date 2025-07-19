import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


# Load API key
load_dotenv()
groq_api_key = os.getenv("groq_api_keys")

# Initialize model & embeddings
model = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)
embeddings = OllamaEmbeddings(model="llama3.2:1b")

# Load Chroma vector DB from hospital PDF
db = Chroma(persist_directory="./chroma_db_my", embedding_function=embeddings)
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 3})

# Streamlit UI
st.set_page_config(page_title="Hospital Assistant", page_icon="🏥")
st.title("🏥 Welcome to Our Hospital Assistant")
st.write("I'm here to answer your questions based on our hospital's documentation.")

# System prompt for hospital
system_prompt = (
    "You are a helpful assistant at a hospital information desk.\n"
    "You MUST answer ONLY using the context provided from the hospital's official document.\n"
    "If the answer is not in the context, say: "
    "'Sorry, I couldn't find that information in our hospital document.'\n"
    "You are STRICTLY PROHIBITED from making up answers or guessing.\n"
    "Use direct quotes if possible.\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# Store chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
query = st.chat_input("How can I assist you today?")

if query:
    st.session_state.chat_history.append(("user", query))

    # DEBUG: show what the retriever is returning
    docs = retriever.get_relevant_documents(query)
    print("\n--- Retrieved Chunks ---")
    for i, doc in enumerate(docs):
        print(f"\nChunk {i + 1}:\n{doc.page_content}\n")

    qa_chain = create_stuff_documents_chain(model, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)
    response = rag_chain.invoke({"input": query})
    answer = response["answer"]

    st.session_state.chat_history.append(("assistant", answer))

# Display the chat
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)


