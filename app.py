import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from agent import Agent  # noqa: E402

st.set_page_config(page_title="JuriRH", page_icon="⚖️")
st.title("⚖️ JuriRH")
st.caption("Assistant sur le Code du travail français, 100 % local. "
           "Information générale, pas un conseil juridique.")


@st.cache_resource(show_spinner="Chargement des modèles...")
def charger_agent():
    return Agent()


def afficher_articles(articles, liens):
    """Articles consultés, avec un lien Légifrance quand il est connu."""
    if not articles:
        return
    items = [f"[{a}]({liens[a]})" if liens.get(a) else a for a in articles]
    st.caption("Articles consultés : " + ", ".join(items))


agent = charger_agent()

if "historique" not in st.session_state:
    st.session_state.historique = []

for msg in st.session_state.historique:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        afficher_articles(msg.get("articles"), msg.get("liens", {}))

question = st.chat_input("Posez votre question sur le Code du travail...")
if question:
    st.session_state.historique.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche des articles..."):
            r = agent.repondre(question, streaming=True)

        if r.get("flux") is not None:
            # Question de droit : la réponse s'écrit en direct
            texte = st.write_stream(r["flux"])
        else:
            # Refus ou calcul : réponse déjà complète
            texte = r["reponse"]
            st.markdown(texte)

        liens = r.get("liens", {})
        afficher_articles(r["articles"], liens)
        # "rag_direct" est interne (simple recherche) : on n'affiche que les vrais outils
        outils = [a["outil"] for a in r["outils"] if a["outil"] != "rag_direct"]
        if outils:
            st.caption("Outils utilisés : " + ", ".join(outils))

    st.session_state.historique.append(
        {"role": "assistant", "content": texte,
         "articles": r["articles"], "liens": liens}
    )