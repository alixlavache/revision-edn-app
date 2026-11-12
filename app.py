import streamlit as st
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

st.set_page_config(page_title="Révision EDN", layout="wide")
st.title("Assistant de Révision EDN")

# --- BARRE LATÉRALE ---
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Clé API Gemini", type="password")
    uploaded_file = st.file_uploader("Uploader un PDF (Cours)", type="pdf")

# --- FONCTIONS RAG ---
def get_pdf_text(pdf_file):
    text = ""
    reader = PyPDF2.PdfReader(pdf_file)
    for page in reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    return chunks

def get_vector_store(chunks, api_key):
    # Utilise le modèle d'embedding gratuit de Google
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain(api_key):
    prompt_template = """
    Tu es un tuteur médical pour un étudiant préparant les EDN.
    Réponds de manière détaillée en te basant STRICTEMENT sur le contexte fourni (qui est un extrait du cours).
    Si la réponse n'est pas dans le contexte, dis simplement "Je ne trouve pas cette information dans le cours fourni".\n\n
    Contexte:\n {context}?\n
    Question: \n{question}\n
    Réponse:
    """
    model = ChatGoogleGenerativeAI(model="gemini-3.8-flash", google_api_key=api_key, temperature=0.1)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# --- LOGIQUE PRINCIPALE ---
if api_key:
    # 1. Traitement du PDF
    if uploaded_file:
        if "last_file" not in st.session_state or st.session_state.last_file != uploaded_file.name:
            with st.spinner("Analyse du document et création de l'index (cela peut prendre un moment pour 400 pages)..."):
                try:
                    raw_text = get_pdf_text(uploaded_file)
                    text_chunks = get_text_chunks(raw_text)
                    
                    vector_store = get_vector_store(text_chunks, api_key)
                    st.session_state.vector_store = vector_store
                    
                    st.session_state.last_file = uploaded_file.name
                    st.session_state.messages = [] # Réinitialise le chat
                    st.success("Cours indexé avec succès ! Posez votre question.")
                except Exception as e:
                    st.error(f"Erreur lors de la lecture du PDF : {e}")

    # 2. Interface de Chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Pose une question sur ce cours..."):
        if "vector_store" not in st.session_state:
            st.warning("Veuillez d'abord uploader un cours.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                try:
                    with st.spinner("Recherche des pages pertinentes..."):
                        # On ne cherche que les 4 morceaux les plus pertinents
                        docs = st.session_state.vector_store.similarity_search(prompt, k=4)
                        chain = get_conversational_chain(api_key)
                        response = chain({"input_documents": docs, "question": prompt}, return_only_outputs=True)
                        
                        st.write(response["output_text"])
                        st.session_state.messages.append({"role": "assistant", "content": response["output_text"]})
                except Exception as e:
                    st.error(f"Erreur API : {e}")
else:
    st.info("Veuillez entrer votre clé API Gemini dans la barre latérale.")