from __future__ import annotations

import json
from pathlib import Path

from optimization import (
    FORMATION_TEMPLATES,
    run_for_team,
    formation_to_slots,
    ObjectiveWeights,
    PlayerScoreConfig,
    SearchConfig,
)

REPO_ROOT = Path(__file__).resolve().parent

# All 18 Bundesliga teams (from Fully_connected_team_networks_with_kpis_and_netmetrics/)
TEAMS = [
    "1._fc_heidenheim_1846",
    "1._fc_koeln",
    "1._fc_union_berlin",
    "1._fsv_mainz_05",
    "bayer_04_leverkusen",
    "borussia_dortmund",
    "borussia_moenchengladbach",
    "eintracht_frankfurt",
    "fc_augsburg",
    "fc_bayern_muenchen",
    "rasenballsport_leipzig",
    "sc_freiburg",
    "sv_darmstadt_98",
    "sv_werder_bremen",
    "tsg_1899_hoffenheim",
    "vfb_stuttgart",
    "vfl_bochum",
    "vfl_wolfsburg",
]

FORMATION = "4-3-3"
WEIGHTS = ObjectiveWeights(w_kpi=0.60, w_net=0.10, w_cohesion=0.30)
SCORE_CFG = PlayerScoreConfig(std_penalty=0.35, min_minutes=200.0)
SEARCH_CFG = SearchConfig(max_iters=2500)


def main():
    slots = formation_to_slots(FORMATION)
    all_results = {}

    for team in TEAMS:
        print(f"\n=== {team} ===")
        try:
            res = run_for_team(
                team_query=team,
                repo_root=REPO_ROOT,
                formation_slots=slots,
                positions_col="positions",
                weights=WEIGHTS,
                score_cfg=SCORE_CFG,
                seed=7,
                search_cfg=SEARCH_CFG,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results[team] = {"error": str(e)}
            continue

        lineup = res["lineup"]
        obj = res["objective"]
        print(f"  resolved: {res['team_resolved_name']}")
        for slot in slots:
            print(f"  {slot:>3}  {lineup[slot]}")
        print(f"  objective: total={obj['total']:.4f} "
              f"kpi={obj['kpi_norm']:.4f} net={obj['net_norm']:.4f} coh={obj['cohesion_norm']:.4f}")

        all_results[team] = {
            "resolved": res["team_resolved_name"],
            "lineup": {s: lineup[s] for s in slots},
            "objective": {
                "total": float(obj["total"]),
                "kpi_norm": float(obj["kpi_norm"]),
                "net_norm": float(obj["net_norm"]),
                "cohesion_norm": float(obj["cohesion_norm"]),
                "kpi_term": float(obj["kpi_term"]),
                "net_term": float(obj["net_term"]),
                "cohesion_raw": float(obj["cohesion_raw"]),
            },
        }

    out_path = REPO_ROOT / "all_teams_lineups.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
