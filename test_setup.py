import ollama
from sentence_transformers import SentenceTransformer

MODELE = "qwen2.5:3b"  # modèle utilisé par le projet (gère les outils)

# Test 1 : LLM local
r = ollama.chat(
    model=MODELE,
    messages=[{"role": "user", "content": "Dis bonjour en une phrase."}],
)
print("LLM OK :", r["message"]["content"])

# Test 2 : embeddings (téléchargement la première fois)
model = SentenceTransformer("intfloat/multilingual-e5-small")
emb = model.encode(["query: licenciement économique"])
print("Embeddings OK, dimension :", emb.shape[1])