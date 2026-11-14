"""
Executa tots els agents de estandard/agents/ sobre train complet i sobre la partició de validació
(darrers 7 dies de cada bloc, com el test). Escriu results/resultats.csv i results/resultats.md.

Ús:  python estandard/run_all.py [nom_agent ...]
"""
import sys, json, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from data import load_hourly, split_val
from evaluate import AGENTS, ast_ok, load_agent, evaluate, oracle_score

OUT = Path(__file__).parent / "results"


def main(names):
    df = load_hourly(); fit, val = split_val(df)
    sets = {"train_complet": df, "validacio": val}
    orc = {k: oracle_score(v) for k, v in sets.items()}
    rows = []
    folders = sorted(p for p in AGENTS.iterdir() if (p / "agent.py").exists())
    if names:
        folders = [f for f in folders if f.name in names]
    for f in folders:
        err = ast_ok(f)
        for sname, sdf in sets.items():
            r = {"agent": f.name, "conjunt": sname, "ast": err or "ok"}
            try:
                r.update(evaluate(load_agent(f), sdf))
                r["gap_oracle"] = round(orc[sname] - r["score"], 3)
            except Exception as e:  # noqa: BLE001
                r["error"] = repr(e)
            print(json.dumps(r, ensure_ascii=False)); rows.append(r)
    OUT.mkdir(exist_ok=True)
    res = pd.DataFrame(rows)
    for s, v in orc.items():
        res = pd.concat([res, pd.DataFrame([{"agent": "_oracle_LP", "conjunt": s, "score": v}])])
    res.to_csv(OUT / "resultats.csv", index=False)
    cols = ["agent", "conjunt", "score", "score_web", "co2_estalviat_pct", "cost_estalviat_pct",
            "infraccions", "gap_oracle", "score_public", "score_private", "cicles_equivalents", "ms_per_step", "ast"]
    tab = res[[c for c in cols if c in res]].sort_values(["conjunt", "score"], ascending=[True, False])
    (OUT / "resultats.md").write_text(tab.to_markdown(index=False))
    print(tab.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
