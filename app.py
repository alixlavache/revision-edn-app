import streamlit as st
import PyPDF2
from google import genai
from google.genai import types

st.set_page_config(page_title="Révision EDN", layout="wide")
st.title("Assistant de Révision EDN")

# --- BARRE LATÉRALE ---
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Clé API Gemini", type="password")
    uploaded_file = st.file_uploader("Uploader un PDF (Cours)", type="pdf")

# --- LOGIQUE PRINCIPALE ---
if api_key:
    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1beta'})

    # 1. Extraction du PDF
    if uploaded_file:
        if "last_file" not in st.session_state or st.session_state.last_file != uploaded_file.name:
            with st.spinner("Extraction du texte en cours..."):
                try:
                    reader = PyPDF2.PdfReader(uploaded_file)
                    texte_extrait = "".join(page.extract_text() + "\n" for page in reader.pages)
                    
                    st.session_state.pdf_text = texte_extrait
                    st.session_state.last_file = uploaded_file.name
                    st.session_state.messages = [] # Réinitialise le chat pour le nouveau cours
                    st.success("Cours chargé avec succès !")
                except Exception as e:
                    st.error(f"Erreur lors de la lecture du PDF : {e}")

    # 2. Interface de Chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Affichage de l'historique
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Champ de saisie
    if prompt := st.chat_input("Pose une question sur ce cours..."):
        if "pdf_text" not in st.session_state:
            st.warning("Veuillez d'abord uploader un cours dans la barre latérale.")
        else:
            # Afficher la question de l'utilisateur
            with st.chat_message("user"):
                st.write(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Interroger Gemini
            with st.chat_message("assistant"):
                try:
                    contexte_medical = (
                        "Tu es un tuteur pour un étudiant en médecine préparant les EDN. "
                        "Réponds STRICTEMENT en te basant sur le cours fourni. "
                        f"\n\nCOURS :\n{st.session_state.pdf_text}"
                    )
                    
                    reponse = client.models.generate_content(
                        model='gemini-1.5-flash',
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=contexte_medical,
                            temperature=0.1
                        )
                    )
                    st.write(reponse.text)
                    st.session_state.messages.append({"role": "assistant", "content": reponse.text})
                except Exception as e:
                    st.error(f"Erreur API : {e}")
else:
    st.info("Veuillez entrer votre clé API Gemini dans la barre latérale.")