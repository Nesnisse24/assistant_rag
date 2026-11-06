import json
import re
import ollama
from rag import JuriRH, MODELE_LLM, REFUS, OPTIONS_LLM, KEEP_ALIVE
from tools import indemnite_licenciement, preavis_licenciement, preavis_demission

MAX_TOURS = 3
OUTILS_CALCUL = ("calculer_indemnite_licenciement", "calculer_preavis")

# Une question est un "calcul" si elle parle d'indemnité/préavis ET contient un chiffre
# (ancienneté, salaire...). Sinon : question de droit pure -> 1 seul appel LLM.
RE_SUJET_CALCUL = re.compile(r"indemnit|pr[ée]avis", re.I)
RE_CHIFFRE = re.compile(r"\d")

SYSTEM_PROMPT = f"""Tu es JuriRH, un assistant spécialisé dans le Code du travail français.

Tu disposes de 2 outils de calcul :
- calculer_indemnite_licenciement : pour calculer une indemnité légale de licenciement.
- calculer_preavis : UNIQUEMENT pour la durée du préavis de licenciement ou de démission.

Règles strictes :
1. Appelle l'outil adapté pour obtenir le résultat. Ne calcule jamais de mémoire.
2. Donne l'ancienneté en années complètes et en mois supplémentaires, sans aucune
   conversion (ex. 15 ans -> anciennete_annees=15 ; 4 ans et 6 mois -> 4 et 6). Si le salaire
   mensuel brut ou l'ancienneté manque, demande-le à l'utilisateur au lieu de l'inventer.
3. Recopie exactement les montants renvoyés par l'outil, sans les modifier et sans refaire le calcul.
4. Si l'outil ne donne pas l'information, réponds exactement : "{REFUS}"
5. Réponds en français, de façon claire et concise.
6. Rappelle que les montants sont indicatifs (une convention collective peut être plus favorable)
   et que tu fournis une information générale, pas un conseil juridique personnalisé."""

OUTILS = [
    {
        "type": "function",
        "function": {
            "name": "calculer_indemnite_licenciement",
            "description": "Calcule l'indemnité légale minimale de licenciement (R1234-2).",
            "parameters": {
                "type": "object",
                "properties": {
                    "anciennete_annees": {"type": "integer",
                                          "description": "Années complètes d'ancienneté (15 ans -> 15)"},
                    "anciennete_mois_en_plus": {"type": "integer",
                                                "description": "Mois en plus des années complètes (0 si aucun)"},
                    "salaire_ref": {"type": "number",
                                    "description": "Salaire de référence mensuel brut en euros"},
                    "inaptitude_professionnelle": {"type": "boolean",
                                                   "description": "True si inaptitude d'origine professionnelle"},
                },
                "required": ["anciennete_annees", "salaire_ref"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculer_preavis",
            "description": "Donne la durée du préavis de licenciement ou de démission (et rien d'autre).",
            "parameters": {
                "type": "object",
                "properties": {
                    "motif": {"type": "string", "enum": ["licenciement", "demission"]},
                    "anciennete_annees": {"type": "integer", "description": "Années complètes d'ancienneté"},
                    "anciennete_mois_en_plus": {"type": "integer",
                                                "description": "Mois en plus des années complètes (0 si aucun)"},
                },
                "required": ["motif"],
            },
        },
    },
]


def formater_calcul(resultat) -> str:
    """Affiche le résultat brut de l'outil, tel quel (clés: valeurs)."""
    if isinstance(resultat, dict):
        return "; ".join(f"{k} : {v}" for k, v in resultat.items())
    return str(resultat)


def est_calcul(question: str) -> bool:
    return bool(RE_SUJET_CALCUL.search(question) and RE_CHIFFRE.search(question))


class Agent:
    def __init__(self):
        self.rag = JuriRH()
        self.rag.prechauffer()   # charge le LLM en mémoire dès le démarrage

    def _executer(self, nom: str, args: dict):
        try:
            if nom in OUTILS_CALCUL:
                # La conversion années -> mois est faite ici, pas par le modèle
                mois = (int(args.get("anciennete_annees", 0)) * 12
                        + int(args.get("anciennete_mois_en_plus", 0)))
                if nom == "calculer_indemnite_licenciement":
                    return indemnite_licenciement(
                        mois,
                        float(args["salaire_ref"]),
                        bool(args.get("inaptitude_professionnelle", False)),
                    )
                if args.get("motif") == "demission":
                    return preavis_demission()
                return preavis_licenciement(mois)
            return {"erreur": f"Outil inconnu : {nom}"}
        except (KeyError, ValueError, TypeError) as e:
            return {"erreur": f"Arguments invalides : {e}"}

    def _liens(self, articles):
        """{numéro d'article: URL Légifrance} pour les articles connus de l'index."""
        liens = {a: self.rag.lien_article(a) for a in articles}
        return {a: u for a, u in liens.items() if u}

    def _rag_direct(self, question: str, passages, streaming: bool):
        """Recherche déjà faite + 1 seul appel LLM (en flux si streaming=True)."""
        articles = sorted({p["meta"]["num"] for p in passages})
        outils = [{"outil": "rag_direct", "args": {}}]
        if streaming:
            return {"reponse": None,
                    "flux": self.rag.repondre_flux(question, passages),
                    "outils": outils, "articles": articles,
                    "liens": self._liens(articles)}
        r = self.rag.repondre(question, passages)
        return {"reponse": r["reponse"], "outils": outils, "articles": articles,
                "liens": self._liens(articles)}

    def repondre(self, question: str, streaming: bool = False):
        """
        Retourne un dict {"reponse", "outils", "articles"}.
        Si streaming=True et qu'il s'agit d'une question de droit, "reponse" vaut None
        et "flux" contient un générateur de texte à afficher avec st.write_stream.
        """
        # Garde-fou question trop éloignée du Code du travail -> refus sans appeler le LLM
        passages = self.rag.rechercher(question)
        if self.rag.hors_sujet(passages):
            return {"reponse": REFUS, "outils": [], "articles": [], "liens": {}}

        # Question de droit pure 1 seul appel LLM, pas de boucle d'outils
        if not est_calcul(question):
            return self._rag_direct(question, passages, streaming)

        # Question de calcul  boucle d'outils (2 outils seulement, prompt court)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        appels = []          # trace des outils utilisés (utile pour l'évaluation)
        dernier_calcul = None
        articles_calcul = set()   # articles renvoyés par les outils de calcul

        for _ in range(MAX_TOURS):
            rep = ollama.chat(model=MODELE_LLM, messages=messages, tools=OUTILS,
                              keep_alive=KEEP_ALIVE, options=OPTIONS_LLM)
            msg = rep.message
            messages.append(msg)

            if not msg.tool_calls:
                # Aucun outil appelé du tout on bascule sur la recherche directe
                if not appels:
                    return self._rag_direct(question, passages, streaming)
                texte = msg.content or REFUS
                if dernier_calcul:
                    texte += ("\n\nRésultat du calcul (source : outil) : "
                              + formater_calcul(dernier_calcul))
                articles = sorted(articles_calcul)
                return {"reponse": texte, "outils": appels, "articles": articles,
                        "liens": self._liens(articles)}

            for call in msg.tool_calls:
                nom, args = call.function.name, dict(call.function.arguments)
                resultat = self._executer(nom, args)
                appels.append({"outil": nom, "args": args})
                if nom in OUTILS_CALCUL and isinstance(resultat, dict) \
                        and "erreur" not in resultat:
                    dernier_calcul = resultat
                    arts = resultat.get("articles") or [resultat.get("article")]
                    articles_calcul.update(a for a in arts if a)
                messages.append({"role": "tool", "tool_name": nom,
                                 "content": json.dumps(resultat, ensure_ascii=False)})

        return {"reponse": REFUS, "outils": appels, "articles": [], "liens": {}}


if __name__ == "__main__":
    agent = Agent()
    for q in [
        "J'ai 5 ans d'ancienneté et un salaire de 2000 euros brut, quelle indemnité de licenciement ?",
        "Quel préavis pour un licenciement après 3 ans dans l'entreprise ?",
        "Combien de jours de congés payés par mois de travail ?",
        "Quelle est la recette de la tarte aux pommes ?",
    ]:
        print("\nQ:", q)
        r = agent.repondre(q)
        print("Outils:", [(a["outil"], a["args"]) for a in r["outils"]])
        print("R:", r["reponse"])