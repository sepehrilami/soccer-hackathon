from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from optimization import (
    run_for_team,
    formation_to_slots,
    ObjectiveWeights,
    PlayerScoreConfig,
    SearchConfig,
)

REPO_ROOT = Path(__file__).resolve().parent
TEAM = "fc_bayern_muenchen"
FORMATION = "4-3-3"

SCORE_CFG = PlayerScoreConfig(std_penalty=0.35, min_minutes=200.0)
SEARCH_CFG = SearchConfig(max_iters=2500)

slots = formation_to_slots(FORMATION)


def run_config(alpha: float, beta: float, gamma: float) -> Dict:
    w = ObjectiveWeights(w_kpi=alpha, w_net=beta, w_cohesion=gamma)
    res = run_for_team(
        team_query=TEAM,
        repo_root=REPO_ROOT,
        formation_slots=slots,
        weights=w,
        score_cfg=SCORE_CFG,
        seed=7,
        search_cfg=SEARCH_CFG,
    )
    obj = res["objective"]
    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
        "lineup": {s: res["lineup"][s] for s in slots},
        "kpi_norm": float(obj["kpi_norm"]),
        "net_norm": float(obj["net_norm"]),
        "coh_norm": float(obj["cohesion_norm"]),
        "kpi_term": float(obj["kpi_term"]),
        "net_term": float(obj["net_term"]),
        "coh_raw":  float(obj["cohesion_raw"]),
        "total":    float(obj["total"]),
    }


def main():
    # 1D sweep along (alpha + gamma = 0.90), beta fixed at 0.10
    gammas = np.round(np.linspace(0.00, 0.90, 10), 2).tolist()
    sweep: List[Dict] = []
    for gamma in gammas:
        alpha = round(0.90 - gamma, 2)
        beta = 0.10
        print(f"sweep: a={alpha:.2f} b={beta:.2f} g={gamma:.2f}")
        sweep.append(run_config(alpha, beta, gamma))

    # Named corner configurations (for the table and narrative)
    corners = {
        "talent_heavy":    (0.90, 0.05, 0.05),
        "default":         (0.60, 0.10, 0.30),
        "balanced":        (0.33, 0.34, 0.33),
        "network_heavy":   (0.05, 0.90, 0.05),
        "chemistry_heavy": (0.05, 0.05, 0.90),
    }
    corner_results: Dict[str, Dict] = {}
    for name, (a, b, g) in corners.items():
        print(f"corner: {name} a={a:.2f} b={b:.2f} g={g:.2f}")
        corner_results[name] = run_config(a, b, g)

    out = {
        "team": TEAM,
        "formation": FORMATION,
        "sweep_axis": "gamma in [0,0.90], beta=0.10, alpha=0.90-gamma",
        "sweep": sweep,
        "corners": corner_results,
    }
    out_path = REPO_ROOT / "bayern_sweep.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
