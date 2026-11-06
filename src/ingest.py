import re
import html
import sys
from pathlib import Path
import chromadb
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

MODELE_EMB = "intfloat/multilingual-e5-small"
DB_PATH = str(Path(__file__).resolve().parent.parent / "chroma_db")
MAX_CARS = 1500
CHEVAUCHEMENT = 150

# python src/ingest.py        -> collection "code_travail" (contexte complet)
# python src/ingest.py court  -> collection "code_travail_court" (2 derniers niveaux)
VARIANTE = sys.argv[1] if len(sys.argv) > 1 else "complet"
COLLECTION = "code_travail" if VARIANTE == "complet" else f"code_travail_{VARIANTE}"


def nettoyer(texte: str) -> str:
    texte = html.unescape(texte or "")
    texte = re.sub(r"\s+", " ", texte)
    return texte.strip()


def contexte_pour_embedding(contexte: str) -> str:
    if VARIANTE != "court":
        return contexte
    niveaux = [n.strip() for n in contexte.split(">") if n.strip()]
    return " > ".join(niveaux[-2:])


def decouper(texte: str):
    if len(texte) <= MAX_CARS:
        return [texte]
    morceaux, debut = [], 0
    while debut < len(texte):
        morceaux.append(texte[debut:debut + MAX_CARS])
        debut += MAX_CARS - CHEVAUCHEMENT
    return morceaux


def main():
    print(f"Variante : {VARIANTE} -> collection {COLLECTION}")
    ds = load_dataset("louisbrulenaudet/code-travail", split="train")

    docs, ids, metas = [], [], []
    vus = set()
    for art in ds:
        if art["etat"] != "VIGUEUR":
            continue
        texte = nettoyer(art["texte"])
        if len(texte) < 20:
            continue

        cle = (art["num"], texte)
        if cle in vus:
            continue
        vus.add(cle)

        contexte = nettoyer(art["fullSectionsTitre"])
        ctx_emb = contexte_pour_embedding(contexte)
        for morceau in decouper(texte):
            docs.append(f"passage: {art['ref']}. {ctx_emb}. {morceau}")
            ids.append(f"passage_{len(ids)}")
            metas.append({
                "num": art["num"],
                "ref": art["ref"],
                "contexte": contexte,
                "legi_id": art["id"],
            })

    print(f"{len(docs)} passages à indexer")

    modele = SentenceTransformer(MODELE_EMB)
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    TAILLE_LOT = 64
    for i in range(0, len(docs), TAILLE_LOT):
        lot = docs[i:i + TAILLE_LOT]
        emb = modele.encode(lot, normalize_embeddings=True).tolist()
        col.add(ids=ids[i:i + TAILLE_LOT], documents=lot,
                embeddings=emb, metadatas=metas[i:i + TAILLE_LOT])
        if (i // TAILLE_LOT) % 20 == 0:
            print(f"  {i + len(lot)}/{len(docs)}")

    print("Indexation terminée :", col.count(), "passages")


if __name__ == "__main__":
    main()