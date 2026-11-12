import streamlit as st
import PyPDF2
from google import genai

st.set_page_config(page_title="Révision EDN", layout="wide")
st.title("Assistant de Révision EDN")

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Clé API Gemini", type="password")
    uploaded_file = st.file_uploader("Uploader un PDF (Cours)", type="pdf")

if api_key:
    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1beta'})
    
    if uploaded_file:
        if "last_file" not in st.session_state or st.session_state.last_file != uploaded_file.name:
            with st.spinner("Extraction des 400 pages (patientez quelques instants)..."):
                try:
                    reader = PyPDF2.PdfReader(uploaded_file)
                    texte_extrait = "".join([page.extract_text() + "\n" for page in reader.pages])
                    st.session_state.pdf_text = texte_extrait
                    st.session_state.last_file = uploaded_file.name
                    st.session_state.messages = []
                    st.success("Cours complet chargé avec succès !")
                except Exception as e:
                    st.error(f"Erreur de lecture : {e}")
                    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
    if prompt := st.chat_input("Pose une question sur ce cours..."):
        if "pdf_text" not in st.session_state:
            st.warning("Veuillez d'abord uploader un cours.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
                
            with st.chat_message("assistant"):
                with st.spinner("Réflexion de l'IA..."):
                    try:
                        contexte = f"Tu es un tuteur pour les EDN. Base-toi STRICTEMENT sur ce cours :\n\n{st.session_state.pdf_text}"
                        response = client.models.generate_content(
                            model='gemini-1.5-pro',
                            contents=[contexte, prompt]
                        )
                        st.write(response.text)
                        st.session_state.messages.append({"role": "assistant", "content": response.text})
                    except Exception as e:
                        st.error(f"Erreur API : {e}")
else:
    st.info("Veuillez entrer votre clé API Gemini dans la barre latérale.")