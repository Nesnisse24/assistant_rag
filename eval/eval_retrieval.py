import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import rag

collection = sys.argv[1] if len(sys.argv) > 1 else "code_travail_court"
rag.COLLECTION = collection
bot = rag.JuriRH()
print(f"Collection testée : {collection}\n")

questions = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))

total = hit5 = hit10 = 0
for q in questions:
    if q["cat"] != "rag":
        continue
    total += 1
    top = [p["meta"]["num"] for p in bot.rechercher(q["question"], k=20)]
    rangs = [top.index(a) + 1 for a in q["attendus"] if a in top]
    meilleur = min(rangs) if rangs else None
    hit5 += bool(meilleur and meilleur <= 5)
    hit10 += bool(meilleur and meilleur <= 10)
    etat = "OK" if meilleur and meilleur <= 5 else "KO"
    print(f"[{etat}] rang {meilleur if meilleur else '>20':>3} | {q['attendus']} | {q['question'][:55]}")

print(f"\nTop 5  : {hit5}/{total}")
print(f"Top 10 : {hit10}/{total}")