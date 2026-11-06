import re
from pathlib import Path
import ollama
import chromadb
from sentence_transformers import SentenceTransformer

MODELE_EMB = "intfloat/multilingual-e5-small"
DB_PATH = str(Path(__file__).resolve().parent.parent / "chroma_db")
URL_LEGIFRANCE = "https://www.legifrance.gouv.fr/codes/article_lc/{}"
COLLECTION = "code_travail_court"
MODELE_LLM = "qwen2.5:3b"   # gemma3:4b ne gère pas les outils dans Ollama
TOP_K = 4                   # 4 articles suffisent et le prompt reste court (avant : 8)
SEUIL_DISTANCE = 0.15       # calibré : droit <= 0,121 ; hors sujet >= 0,176
MAX_CAR_ARTICLE = 1700      # > MAX_CARS d'ingest.py (1500) + préfixe : un passage n'est plus coupé

# IMPORTANT : garder exactement les mêmes options (surtout num_ctx) pour TOUS les appels,
# sinon Ollama recharge le modèle quand num_ctx change.
OPTIONS_LLM = {"temperature": 0, "num_ctx": 4096, "num_predict": 400}
KEEP_ALIVE = "30m"          # garde le modèle en mémoire entre deux questions

REFUS = "Je ne trouve pas cette information dans les articles du Code du travail dont je dispose."

SYSTEM_PROMPT = f"""Tu es JuriRH, un assistant spécialisé dans le Code du travail français.

Règles strictes :
1. Réponds UNIQUEMENT à partir des extraits fournis dans <sources>.
2. Cite chaque affirmation avec le numéro d'article entre crochets, par exemple [L1234-1].
3. Si les extraits ne contiennent pas l'information nécessaire, réponds exactement :
   "{REFUS}"
4. N'invente jamais de numéro d'article ni de contenu.
5. Réponds en français, de façon claire, structurée et concise.
6. Tu fournis une information générale, pas un conseil juridique personnalisé."""

SIGLES = {
    r"\bCDD\b": "contrat de travail à durée déterminée",
    r"\bCDI\b": "contrat de travail à durée indéterminée",
    r"\bRTT\b": "réduction du temps de travail",
    r"\bCSE\b": "comité social et économique",
    r"\bCP\b": "congés payés",
}


def developper_sigles(question: str) -> str:
    for motif, texte in SIGLES.items():
        question = re.sub(motif, texte, question)
    return question


def _nettoyer(texte: str) -> str:
    """Retire le préfixe 'passage: ' et tronque l'article."""
    texte = texte.replace("passage: ", "", 1)
    if len(texte) > MAX_CAR_ARTICLE:
        texte = texte[:MAX_CAR_ARTICLE].rstrip() + "…"
    return texte


class JuriRH:
    def __init__(self):
        self.modele = SentenceTransformer(MODELE_EMB)
        self.col = chromadb.PersistentClient(path=DB_PATH).get_collection(COLLECTION)
        self._liens = {}

    def prechauffer(self):
        """Charge le LLM en mémoire au démarrage pour accélérer la 1re question."""
        try:
            ollama.chat(
                model=MODELE_LLM,
                messages=[{"role": "user", "content": "ok"}],
                keep_alive=KEEP_ALIVE,
                options={**OPTIONS_LLM, "num_predict": 1},
            )
        except Exception:
            pass  # ce n'est qu'une optimisation, on ignore les erreurs

    def rechercher(self, question: str, k: int = TOP_K):
        question = developper_sigles(question)
        emb = self.modele.encode(
            [f"query: {question}"], normalize_embeddings=True
        ).tolist()
        res = self.col.query(
            query_embeddings=emb,
            n_results=k * 3,
            include=["documents", "metadatas", "distances"],
        )
        vus, resultats = set(), []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            if meta["num"] in vus:
                continue
            vus.add(meta["num"])
            resultats.append({"texte": doc, "meta": meta, "distance": dist})
            if len(resultats) == k:
                break
        return resultats

    def lien_article(self, num: str):
        """URL Légifrance d'un article (via l'identifiant LEGI stocké à l'indexation)."""
        if num not in self._liens:
            metas = self.col.get(where={"num": num}, limit=1, include=["metadatas"])["metadatas"]
            self._liens[num] = URL_LEGIFRANCE.format(metas[0]["legi_id"]) if metas else None
        return self._liens[num]

    @staticmethod
    def hors_sujet(passages) -> bool:
        return not passages or passages[0]["distance"] > SEUIL_DISTANCE

    @staticmethod
    def sources_de(passages):
        return [
            {"num": p["meta"]["num"], "contexte": p["meta"]["contexte"],
             "distance": round(p["distance"], 3)}
            for p in passages
        ]

    def _messages(self, question: str, passages):
        sources_txt = "\n\n".join(
            f"[{p['meta']['num']}] {_nettoyer(p['texte'])}" for p in passages
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"<sources>\n{sources_txt}\n</sources>\n\nQuestion : {question}",
            },
        ]

    def repondre(self, question: str, passages=None):
        """Réponse complète (sans streaming). `passages` évite de refaire la recherche."""
        if passages is None:
            passages = self.rechercher(question)

        # Garde-fou : si le meilleur passage est trop éloigné, on refuse sans appeler le LLM
        if self.hors_sujet(passages):
            return {"reponse": REFUS, "sources": []}

        reponse = ollama.chat(
            model=MODELE_LLM,
            messages=self._messages(question, passages),
            keep_alive=KEEP_ALIVE,
            options=OPTIONS_LLM,
        )
        return {
            "reponse": reponse["message"]["content"],
            "sources": self.sources_de(passages),
        }

    def repondre_flux(self, question: str, passages):
        """Générateur de morceaux de texte (pour l'affichage en direct)."""
        flux = ollama.chat(
            model=MODELE_LLM,
            messages=self._messages(question, passages),
            keep_alive=KEEP_ALIVE,
            options=OPTIONS_LLM,
            stream=True,
        )
        for morceau in flux:
            yield morceau["message"]["content"]


if __name__ == "__main__":
    bot = JuriRH()
    for q in [
        "Quelle est la durée du préavis en cas de démission ?",
        "Combien de jours de congés payés par mois de travail ?",
        "Quelle est la recette de la tarte aux pommes ?",
    ]:
        print("\nQ:", q)
        r = bot.repondre(q)
        print("R:", r["reponse"])
        print("Sources:", [(s["num"], s["distance"]) for s in r["sources"]])