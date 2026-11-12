"""
Interface web Streamlit du chatbot Lockall (démonstration).

Cette app réutilise le code existant :
- chatbot_lockall_propre.py : prompt, construction de la requête et du contexte
- vectorize_data.py        : construction de l'index FAISS (si le dossier est absent)

Lancer en local :  streamlit run app.py
"""

import os

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings


# --------------------------------------------------------------------------
# Configuration et secrets
# --------------------------------------------------------------------------
def get_secret(name, default=None):
    """Lit un secret : secrets de la plateforme (en ligne), sinon variable
    d'environnement (.env en local)."""
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name, default)


# La clé est lue AVANT d'importer le chatbot (qui appelle load_dotenv()).
_api_key = get_secret("MISTRAL_API_KEY")
if _api_key:
    os.environ["MISTRAL_API_KEY"] = _api_key

from chatbot_lockall_propre import (  # noqa: E402
    FAISS_INDEX_DIR,
    MAX_HISTORY_MESSAGES,
    SYSTEM_PROMPT,
    TOP_K,
    build_rag_context,
    build_search_query,
)

MAX_QUESTIONS = int(get_secret("MAX_QUESTIONS", 25))
ACCESS_CODE = get_secret("ACCESS_CODE")
MAX_INPUT_CHARS = 500
WELCOME = (
    "Bonjour ! Je suis l'assistant client virtuel de Lockall. "
    "Comment puis-je vous aider aujourd'hui ?"
)

st.set_page_config(page_title="Support client Lockall (démo)", page_icon="🏢")


# --------------------------------------------------------------------------
# Ressources chargées une seule fois (partagées entre les visiteurs)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Chargement de la base de connaissances...")
def get_vector_store():
    if os.path.isdir(FAISS_INDEX_DIR):
        embeddings = MistralAIEmbeddings(model="mistral-embed")
        return FAISS.load_local(
            FAISS_INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True,  # index créé par nous-mêmes
        )
    # Pas d'index dans le dépôt : on le reconstruit (quelques fiches seulement).
    from vectorize_data import create_and_save_vector_store

    return create_and_save_vector_store()


@st.cache_resource
def get_llm():
    return ChatMistralAI(
        model="mistral-medium-latest",
        temperature=0.1,
        max_tokens=600,
    )


# --------------------------------------------------------------------------
# État de la session (une conversation par visiteur)
# --------------------------------------------------------------------------
def reset_conversation():
    st.session_state.history = [SystemMessage(content=SYSTEM_PROMPT)]
    st.session_state.display = [{"role": "assistant", "content": WELCOME}]
    st.session_state.count = 0


if "history" not in st.session_state:
    reset_conversation()
if "unlocked" not in st.session_state:
    st.session_state.unlocked = not ACCESS_CODE


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
st.title("🏢 Support client Lockall")
st.caption(
    "Projet de démonstration non officiel, non affilié à Lockall. "
    "Pour une demande réelle, utilisez les contacts officiels ou la "
    "[FAQ Lockall](https://www.lockall.fr/aide/)."
)

with st.sidebar:
    st.subheader("Conversation")
    remaining = max(MAX_QUESTIONS - st.session_state.count, 0)
    st.write(f"Questions restantes : **{remaining}** / {MAX_QUESTIONS}")
    if st.button("Nouvelle conversation"):
        reset_conversation()
        st.rerun()

if not _api_key:
    st.error(
        "La clé MISTRAL_API_KEY est introuvable. Ajoute-la dans les secrets de "
        "la plateforme (ou dans ton fichier .env en local)."
    )
    st.stop()

if not st.session_state.unlocked:
    code = st.text_input("Code d'accès", type="password")
    if code:
        if code == ACCESS_CODE:
            st.session_state.unlocked = True
            st.rerun()
        else:
            st.error("Code incorrect.")
    st.stop()

for message in st.session_state.display:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.count >= MAX_QUESTIONS:
    st.info(
        "Limite de questions atteinte pour cette démonstration. "
        "Cliquez sur « Nouvelle conversation » dans le menu de gauche pour recommencer."
    )
    st.stop()

user_input = st.chat_input("Écrivez votre message...")

if user_input:
    user_input = user_input.strip()[:MAX_INPUT_CHARS]

    st.session_state.display.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Je réfléchis..."):
                history = st.session_state.history
                vector_store = get_vector_store()
                search_query = build_search_query(history, user_input)
                docs = vector_store.similarity_search(search_query, k=TOP_K)

                augmented = (
                    f"{build_rag_context(docs)}\n\n"
                    f"Question du client : {user_input}"
                )
                response = get_llm().invoke(history + [HumanMessage(content=augmented)])
                answer = response.content
        except Exception:
            answer = (
                "Désolé, un problème technique est survenu. "
                "Pouvez-vous réessayer dans quelques instants ?"
            )
            st.session_state.display.append({"role": "assistant", "content": answer})
            st.markdown(answer)
            st.stop()

        st.markdown(answer)

    # Historique léger : question brute, sans le contexte RAG (comme dans le CLI).
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=answer))
    if len(history) > MAX_HISTORY_MESSAGES + 1:
        history[:] = [history[0]] + history[-MAX_HISTORY_MESSAGES:]

    st.session_state.display.append({"role": "assistant", "content": answer})
    st.session_state.count += 1
    st.rerun()
