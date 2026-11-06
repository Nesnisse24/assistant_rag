import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import rag
import agent as agent_mod

# Accepte "[L1237-1]", "L. 1237-1", "article L1237-1"
CITATION = re.compile(r"\b([LRD])\.?\s?(\d{4}(?:-\d+)+)")
ARTICLES_OUTILS = {"L1234-1", "L1234-9", "R1234-2", "L1226-14", "L1237-1"}


def norm(texte: str) -> str:
    return re.sub(r"[\s\u00a0\u202f]", "", texte).lower()


def pct(n: int, total: int) -> str:
    return f"{n}/{total} ({100 * n / total:.0f} %)" if total else "n/a"


def evaluer(agent, q):
    t0 = time.time()
    r = agent.repondre(q["question"])
    duree = time.time() - t0

    texte = r["reponse"] or ""
    refus = rag.REFUS in texte
    cites = {f"{a}{b}" for a, b in CITATION.findall(texte)}
    outils = [a["outil"] for a in r["outils"]]
    passages = agent.rag.rechercher(q["question"])
    top = [p["meta"]["num"] for p in passages]
    dist = passages[0]["distance"] if passages else 1.0
    res = {"cat": q["cat"], "question": q["question"], "reponse": texte,
           "outils": outils, "cites": sorted(cites), "refus": refus,
           "refus_par_seuil": dist > rag.SEUIL_DISTANCE,
           "distance_top1": round(dist, 3), "duree_s": round(duree, 1)}

    if q["cat"] == "rag":
        allowed = set(r["articles"]) | set(top) | ARTICLES_OUTILS
        res["hit5"] = any(a in top[:5] for a in q["attendus"])
        res["refus_a_tort"] = refus
        res["article_attendu_cite"] = any(a in cites for a in q["attendus"])
        res["citations_valides"] = bool(cites) and cites <= allowed
        res["ok"] = res["hit5"] and not refus and res["citations_valides"]
    elif q["cat"] == "calcul":
        res["bon_outil"] = q["outil"] in outils
        res["bonne_valeur"] = any(v in norm(texte) for v in q["contient"])
        res["ok"] = res["bon_outil"] and res["bonne_valeur"]
    else:
        res["ok"] = refus
    return res


def erreur(q, e):
    base = {"cat": q["cat"], "question": q["question"], "reponse": f"ERREUR: {e}",
            "outils": [], "cites": [], "refus": False, "refus_par_seuil": False,
            "distance_top1": None, "duree_s": 0, "ok": False, "erreur": True}
    if q["cat"] == "rag":
        base.update(hit5=False, refus_a_tort=False, article_attendu_cite=False,
                    citations_valides=False)
    elif q["cat"] == "calcul":
        base.update(bon_outil=False, bonne_valeur=False)
    return base


def synthese(modele, resultats):
    rag_r = [x for x in resultats if x["cat"] == "rag"]
    cal_r = [x for x in resultats if x["cat"] == "calcul"]
    hs_r = [x for x in resultats if x["cat"] == "hors_sujet"]
    nb_err = sum(1 for x in resultats if x.get("erreur"))

    print("\n" + "=" * 60)
    print(f"SYNTHESE - modele {modele} ({len(resultats)} questions, {nb_err} erreurs)")
    print("=" * 60)
    print(f"Droit ({len(rag_r)}) :")
    print("  Article attendu dans le top 5     :", pct(sum(x["hit5"] for x in rag_r), len(rag_r)))
    print("  Refus a tort                      :", pct(sum(x["refus_a_tort"] for x in rag_r), len(rag_r)))
    print("  Article attendu cite              :", pct(sum(x["article_attendu_cite"] for x in rag_r), len(rag_r)))
    print("  Citations valides (aucune inventee):", pct(sum(x["citations_valides"] for x in rag_r), len(rag_r)))
    print("  Reponses OK (global)              :", pct(sum(x["ok"] for x in rag_r), len(rag_r)))
    print(f"Calculs ({len(cal_r)}) :")
    print("  Bon outil appele                  :", pct(sum(x["bon_outil"] for x in cal_r), len(cal_r)))
    print("  Bonne valeur dans la reponse      :", pct(sum(x["bonne_valeur"] for x in cal_r), len(cal_r)))
    print(f"Hors sujet ({len(hs_r)}) :")
    print("  Refus corrects                    :", pct(sum(x["ok"] for x in hs_r), len(hs_r)))
    print("    dont bloques par le seuil       :", pct(sum(x["refus_par_seuil"] for x in hs_r), len(hs_r)))
    print("    refuses par le LLM              :",
          pct(sum(1 for x in hs_r if x["ok"] and not x["refus_par_seuil"]), len(hs_r)))
    if resultats:
        print(f"Temps moyen par question          : {sum(x['duree_s'] for x in resultats) / len(resultats):.1f} s")


def main():
    modele = sys.argv[1] if len(sys.argv) > 1 else rag.MODELE_LLM
    rag.MODELE_LLM = modele
    agent_mod.MODELE_LLM = modele
    print(f"Evaluation avec le modele : {modele}\n")

    questions = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
    sortie = ROOT / "eval" / f"resultats_{modele.replace(':', '_')}.json"
    agent = agent_mod.Agent()

    # Reprise : on recharge ce qui a deja ete fait
    resultats = []
    if sortie.exists():
        resultats = json.loads(sortie.read_text(encoding="utf-8"))
        print(f"Reprise : {len(resultats)} questions deja traitees\n")
    faites = {x["question"] for x in resultats}

    for q in questions:
        if q["question"] in faites:
            continue
        try:
            res = evaluer(agent, q)
        except Exception as e:
            res = erreur(q, e)
        resultats.append(res)
        sortie.write_text(json.dumps(resultats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{'OK' if res['ok'] else 'KO'}] {q['cat']:10} {q['question'][:60]}"
              f"  ({res['duree_s']:.0f}s, outils: {res['outils']})")
        if res.get("erreur"):
            print("     ", res["reponse"][:150])

    synthese(modele, resultats)
    print(f"\nResultats detailles : {sortie}")


if __name__ == "__main__":
    main()