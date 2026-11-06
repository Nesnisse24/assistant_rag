import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import rag

bot = rag.JuriRH()
questions = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))

dist = {"rag": [], "calcul": [], "hors_sujet": []}
for q in questions:
    d = bot.rechercher(q["question"], k=1)[0]["distance"]
    dist[q["cat"]].append(d)
    print(f"{q['cat']:10} {d:.3f}  {q['question'][:60]}")

print()
for cat, valeurs in dist.items():
    print(f"{cat:10} min {min(valeurs):.3f} | max {max(valeurs):.3f}")