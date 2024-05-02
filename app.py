from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import VectorDBQA
from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader
import streamlit as st

st.title('Arabic RAG App')

def init_rag():
    with st.status('Getting things ready'):
        loader = DirectoryLoader('data/pdf1', glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
        vectordb = Chroma.from_documents(texts, embeddings)

        qa = VectorDBQA.from_chain_type(llm=OpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"]), chain_type="stuff", vectorstore=vectordb)
        
        return qa

def run_query(query):
    return st.session_state['qa'].run(query)

if 'qa' not in st.session_state:
    st.session_state['qa'] = init_rag()

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if query := st.chat_input("Enter query"):
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    response = run_query(query)

    with st.chat_message("assistant"):
        with st.spinner('Thinking...'):
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

if st.button("Clear history", type="primary"):
    st.session_state.messages = []