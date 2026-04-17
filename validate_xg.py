"""xG validation: compare network-optimized lineups against observed lineups.

Fits a Ridge regression predicting team xG from player dummies + team fixed
effects, then scores each team's network-optimized starting XI against the
empirical distribution of observed match-day xGs.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import mean_squared_error, r2_score

from optimization import (
    FORMATION_TEMPLATES,
    default_kpis,
    default_network_metrics,
    find_team_csvs,
    optimize_lineup,
)
from scoring import ObjectiveWeights, PlayerScoreConfig
from cohesion import CohesionConfig
from optimizer_engine import SearchConfig, ObjectiveConfig


# Map of optimizer filename stem -> xG data team_name
TEAM_NAME_MAP: Dict[str, str] = {
    "1._fc_heidenheim_1846":       "1. FC Heidenheim 1846",
    "1._fc_koeln":                 "1. FC Köln",
    "1._fc_union_berlin":          "1. FC Union Berlin",
    "1._fsv_mainz_05":             "1. FSV Mainz 05",
    "bayer_04_leverkusen":         "Bayer 04 Leverkusen",
    "borussia_dortmund":           "Borussia Dortmund",
    "borussia_moenchengladbach":   "Borussia Mönchengladbach",
    "eintracht_frankfurt":         "Eintracht Frankfurt",
    "fc_augsburg":                 "FC Augsburg",
    "fc_bayern_muenchen":          "FC Bayern München",
    "rasenballsport_leipzig":      "RasenBallsport Leipzig",
    "sc_freiburg":                 "SC Freiburg",
    "sv_darmstadt_98":             "SV Darmstadt 98",
    "sv_werder_bremen":            "SV Werder Bremen",
    "tsg_1899_hoffenheim":         "TSG 1899 Hoffenheim",
    "vfb_stuttgart":               "VfB Stuttgart",
    "vfl_bochum":                  "VfL Bochum",
    "vfl_wolfsburg":               "VfL Wolfsburg",
}


def parse_player_list(x):
    if isinstance(x, list):
        return [str(p).strip() for p in x]
    if pd.isna(x):
        return []
    try:
        out = ast.literal_eval(str(x).strip())
        if isinstance(out, list):
            return [str(p).strip() for p in out]
    except Exception:
        pass
    return []


def load_xg_data(path: str = "match_team_lineups_xg.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["xi_list"] = df["starting_11_players"].apply(parse_player_list)
    df = df[(df["xi_list"].apply(len) == 11) & df["team_xg"].notna()].copy()
    return df


def build_design_matrix(df: pd.DataFrame, include_opponent_fe: bool = False):
    mlb = MultiLabelBinarizer(sparse_output=False)
    X_players = mlb.fit_transform(df["xi_list"])
    player_cols = [f"p::{p}" for p in mlb.classes_]
    X_players = pd.DataFrame(X_players, columns=player_cols, index=df.index)
    X_team = pd.get_dummies(df["team_name"], prefix="team", drop_first=True)
    parts = [X_players, X_team.astype(int)]
    if include_opponent_fe:
        if "opponent_name" not in df.columns:
            raise ValueError("Column 'opponent_name' is required when include_opponent_fe=True")
        X_opp = pd.get_dummies(df["opponent_name"], prefix="opp", drop_first=True)
        parts.append(X_opp.astype(int))
    X = pd.concat(parts, axis=1).astype(float)
    y = df["team_xg"].astype(float)
    return X, y, player_cols, mlb


def fit_ridge_cv(X, y, alphas=(0.1, 0.3, 1.0, 3.0, 10.0, 30.0), seed: int = 7):
    """5-fold CV over alpha; return best alpha and out-of-fold metrics."""
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    best_alpha, best_mean_rmse = None, np.inf
    per_alpha = {}
    for alpha in alphas:
        rmses, r2s = [], []
        for tr, te in kf.split(X):
            model = Ridge(alpha=alpha, fit_intercept=True, random_state=seed)
            model.fit(X.iloc[tr], y.iloc[tr])
            pred = model.predict(X.iloc[te])
            rmses.append(mean_squared_error(y.iloc[te], pred, squared=False))
            r2s.append(r2_score(y.iloc[te], pred))
        m_rmse, m_r2 = float(np.mean(rmses)), float(np.mean(r2s))
        per_alpha[alpha] = {"rmse": m_rmse, "r2": m_r2}
        if m_rmse < best_mean_rmse:
            best_mean_rmse = m_rmse
            best_alpha = alpha

    final = Ridge(alpha=best_alpha, fit_intercept=True, random_state=seed).fit(X, y)
    return final, best_alpha, per_alpha


def build_feature_row(team_name: str, xi: List[str], X_cols, player_cols) -> pd.DataFrame:
    """Build a 1-row design matrix for prediction.

    Opponent dummies (if present) are left at 0 — i.e. predictions are made
    against the baseline opponent. This guarantees that optimized and observed
    XIs are compared on equal footing (same opponent assumption).
    """
    xi_set = set([str(p).strip() for p in xi])
    row = pd.DataFrame([np.zeros(len(X_cols))], columns=X_cols)
    for col in player_cols:
        pname = col.replace("p::", "")
        if pname in xi_set:
            row.loc[0, col] = 1
    tcol = f"team_{team_name}"
    if tcol in X_cols:
        row.loc[0, tcol] = 1
    # Note: any opp_* columns are intentionally left at 0.
    return row


FORMATION_PREFERENCE = ["4-3-3", "4-2-3-1", "3-4-3", "4-1-4-1", "3-5-2", "4-4-2"]


def run_optimizer_for_team(
    repo_root: str,
    team_file: str,
    formation: str = "4-3-3",
    seed: int = 7,
    formations_to_try: List[str] = None,
) -> Tuple[Dict, str]:
    """Run optimizer; fall back across formations if the preferred one is infeasible.

    Returns (result_dict, chosen_formation).
    """
    paths = find_team_csvs(repo_root=repo_root, team_query=team_file)
    nodes = pd.read_csv(paths["nodes_csv"])
    edges = pd.read_csv(paths["edges_csv"])

    kpis = default_kpis(nodes)
    net_metrics = default_network_metrics(nodes)

    weights = ObjectiveWeights(w_kpi=0.55, w_net=0.25, w_cohesion=0.20)
    score_cfg = PlayerScoreConfig(
        std_penalty=0.35, min_minutes=600.0, reliability_floor=0.25
    )
    cohesion_cfg = CohesionConfig(mode="arithmetic_mean")
    obj_cfg = ObjectiveConfig(component_norm="sigmoid", cohesion_norm="edge_max")
    search_cfg = SearchConfig(max_iters=3000, candidate_samples=20, tabu_tenure=50)

    if formations_to_try is None:
        formations_to_try = [formation] + [f for f in FORMATION_PREFERENCE if f != formation]

    last_err = None
    for fmt in formations_to_try:
        try:
            result = optimize_lineup(
                nodes=nodes,
                edges=edges,
                formation_slots=FORMATION_TEMPLATES[fmt],
                kpis=kpis,
                mobility_metrics=net_metrics,
                weights=weights,
                score_cfg=score_cfg,
                cohesion_cfg=cohesion_cfg,
                objective_cfg=obj_cfg,
                search_cfg=search_cfg,
                seed=seed,
                position_group_z=True,
            )
            return result, fmt
        except ValueError as e:
            last_err = e
            continue
    raise last_err if last_err else RuntimeError("No formation feasible")


def fit_and_evaluate(
    df: pd.DataFrame,
    include_opponent_fe: bool,
    seed: int,
    spec_name: str,
):
    """Build design matrix, fit Ridge with CV, return fitted model and metadata."""
    X, y, player_cols, mlb = build_design_matrix(df, include_opponent_fe=include_opponent_fe)
    n_team_fe = sum(c.startswith("team_") for c in X.columns)
    n_opp_fe = sum(c.startswith("opp_") for c in X.columns)
    print("\n" + "-" * 78)
    print(f"Specification: {spec_name}")
    print("-" * 78)
    print(f"  Observations:     {len(df)}")
    print(f"  Unique players:   {len(player_cols)}")
    print(f"  Team FEs:         {n_team_fe}")
    print(f"  Opponent FEs:     {n_opp_fe}")
    print(f"  Design matrix:    {X.shape}")

    ridge, best_alpha, per_alpha = fit_ridge_cv(X, y, seed=seed)
    print("\n  5-fold CV over alpha:")
    for a, m in per_alpha.items():
        tag = " [best]" if a == best_alpha else ""
        print(f"    alpha={a:6.2f}   RMSE={m['rmse']:.4f}   R^2={m['r2']:+.4f}{tag}")
    print(f"  Best alpha:       {best_alpha}")
    print(f"  Best CV RMSE:     {per_alpha[best_alpha]['rmse']:.4f}")
    print(f"  Best CV R^2:      {per_alpha[best_alpha]['r2']:+.4f}")

    return {
        "spec_name": spec_name,
        "include_opponent_fe": include_opponent_fe,
        "ridge": ridge,
        "X_columns": list(X.columns),
        "y": y,
        "player_cols": player_cols,
        "mlb": mlb,
        "best_alpha": best_alpha,
        "per_alpha": per_alpha,
        "n_players": len(player_cols),
        "n_team_fe": n_team_fe,
        "n_opp_fe": n_opp_fe,
        "design_shape": list(X.shape),
    }


def score_team(
    spec: dict,
    team_name: str,
    xi_names: List[str],
    df: pd.DataFrame,
) -> dict:
    """Compute predicted xG for an XI under a fitted spec and summarise
    its position within the team's observed-XI distribution."""
    ridge = spec["ridge"]
    X_cols = spec["X_columns"]
    player_cols = spec["player_cols"]
    mlb = spec["mlb"]

    x_row = build_feature_row(team_name, xi_names, X_cols, player_cols)
    pred_xg = float(ridge.predict(x_row)[0])

    team_obs_rows = df[df["team_name"] == team_name]
    obs_preds = []
    for xi_obs in team_obs_rows["xi_list"].tolist():
        obs_preds.append(float(ridge.predict(
            build_feature_row(team_name, xi_obs, X_cols, player_cols)
        )[0]))
    obs_pred_mean = float(np.mean(obs_preds))
    obs_pred_max = float(np.max(obs_preds))
    rank_among_obs = int((np.array(obs_preds) < pred_xg).sum())

    team_xgs = team_obs_rows["team_xg"].values
    team_mean_obs = float(np.mean(team_xgs))
    team_median_obs = float(np.median(team_xgs))
    team_max_obs = float(np.max(team_xgs))
    percentile = float((team_xgs < pred_xg).mean() * 100)

    coverage = sum(1 for p in xi_names if p in set(mlb.classes_))

    return {
        "pred_xg": pred_xg,
        "obs_pred_mean": obs_pred_mean,
        "obs_pred_max": obs_pred_max,
        "rank_among_obs": rank_among_obs,
        "n_obs": len(obs_preds),
        "team_mean_obs": team_mean_obs,
        "team_median_obs": team_median_obs,
        "team_max_obs": team_max_obs,
        "percentile_vs_obs": percentile,
        "coverage": coverage,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", default=".")
    ap.add_argument("--formation", default="4-3-3")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--output", default="xg_validation_results.json")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")

    df = load_xg_data(os.path.join(args.repo_root, "match_team_lineups_xg.csv"))

    print("=" * 78)
    print("STEP 1 — Fitting Ridge regressions (two specifications)")
    print("=" * 78)
    spec_base = fit_and_evaluate(df, include_opponent_fe=False, seed=args.seed,
                                 spec_name="M1: players + team FE")
    spec_full = fit_and_evaluate(df, include_opponent_fe=True, seed=args.seed,
                                 spec_name="M2: players + team FE + opponent FE")

    print("\n" + "-" * 78)
    print("Naive baseline (team-mean predictor)")
    print("-" * 78)
    team_mean = df.groupby("team_name")["team_xg"].transform("mean")
    baseline_rmse = float(np.sqrt(((df["team_xg"] - team_mean) ** 2).mean()))
    baseline_r2 = float(r2_score(df["team_xg"], team_mean))
    print(f"  RMSE={baseline_rmse:.4f}  R^2={baseline_r2:+.4f}")

    print("\n" + "=" * 78)
    print("STEP 2 — Running optimizer and scoring optimized XI under BOTH specs")
    print("=" * 78)
    rows = []
    for stem, xg_name in TEAM_NAME_MAP.items():
        try:
            result, chosen_fmt = run_optimizer_for_team(
                args.repo_root, stem, formation=args.formation, seed=args.seed
            )
        except Exception as e:
            print(f"  [{stem}] optimizer FAILED: {e}")
            continue

        xi_names = list(result["lineup"].values())
        s1 = score_team(spec_base, xg_name, xi_names, df)
        s2 = score_team(spec_full, xg_name, xi_names, df)

        rows.append({
            "team_file": stem,
            "team_name": xg_name,
            "formation": chosen_fmt,
            "optimal_xi": xi_names,
            "coverage_known_players": s1["coverage"],
            # Model 1: team FE only
            "m1_pred_xg_optimized": s1["pred_xg"],
            "m1_pred_xg_obs_mean": s1["obs_pred_mean"],
            "m1_pred_xg_obs_max": s1["obs_pred_max"],
            "m1_rank_among_observed_xis": s1["rank_among_obs"],
            # Model 2: team FE + opponent FE
            "m2_pred_xg_optimized": s2["pred_xg"],
            "m2_pred_xg_obs_mean": s2["obs_pred_mean"],
            "m2_pred_xg_obs_max": s2["obs_pred_max"],
            "m2_rank_among_observed_xis": s2["rank_among_obs"],
            # Shared stats
            "n_observed_xis": s1["n_obs"],
            "team_mean_observed_xg": s1["team_mean_obs"],
            "team_max_observed_xg": s1["team_max_obs"],
            "optimizer_objective": result["objective"]["total"],
            "kpi_term": result["objective"]["kpi_term"],
            "net_term": result["objective"]["net_term"],
            "cohesion_raw": result["objective"]["cohesion_raw"],
        })

        print(
            f"  [{xg_name:32s}] {chosen_fmt:8s}  "
            f"M1: pred={s1['pred_xg']:.3f} obs_mean={s1['obs_pred_mean']:.3f} "
            f"rank={s1['rank_among_obs']}/{s1['n_obs']}   "
            f"M2: pred={s2['pred_xg']:.3f} obs_mean={s2['obs_pred_mean']:.3f} "
            f"rank={s2['rank_among_obs']}/{s2['n_obs']}"
        )

    print("\n" + "=" * 78)
    print("STEP 3 — Aggregate summary (both specifications)")
    print("=" * 78)
    if not rows:
        print("  No teams successfully evaluated.")
        return
    res_df = pd.DataFrame(rows)

    def _summarise(prefix: str, label: str):
        pred = res_df[f"{prefix}_pred_xg_optimized"]
        obs_mean = res_df[f"{prefix}_pred_xg_obs_mean"]
        obs_max = res_df[f"{prefix}_pred_xg_obs_max"]
        rank = res_df[f"{prefix}_rank_among_observed_xis"]
        n_obs = res_df["n_observed_xis"]
        uplift = pred - obs_mean
        n_above_mean = int((uplift > 0).sum())
        n_above_max = int((pred > obs_max).sum())
        rank_frac = (rank / n_obs).mean()
        print(f"\n  {label}:")
        print(f"    Mean uplift over obs-pred mean:    {uplift.mean():+.4f} xG")
        print(f"    Teams above obs-pred mean:         {n_above_mean} / {len(res_df)}")
        print(f"    Teams above obs-pred max:          {n_above_max} / {len(res_df)}")
        print(f"    Mean rank fraction among obs XIs:  {rank_frac:.3f}  (1.0 = highest)")
        return dict(
            mean_uplift=float(uplift.mean()),
            median_uplift=float(uplift.median()),
            n_above_mean=n_above_mean,
            n_above_max=n_above_max,
            mean_rank_fraction=float(rank_frac),
        )

    print(f"  Teams evaluated:                     {len(res_df)}")
    sum1 = _summarise("m1", "M1 — players + team FE")
    sum2 = _summarise("m2", "M2 — players + team FE + opponent FE")

    print("\n  Agreement between the two specifications:")
    rho = res_df["m1_pred_xg_optimized"].corr(res_df["m2_pred_xg_optimized"])
    rho_rank = res_df[["m1_rank_among_observed_xis", "m2_rank_among_observed_xis"]].corr().iloc[0, 1]
    print(f"    Pearson(pred_xG): {rho:+.3f}")
    print(f"    Pearson(rank):    {rho_rank:+.3f}")

    def _pack(spec: dict) -> dict:
        return {
            "n_observations": int(len(df)),
            "n_players": spec["n_players"],
            "n_team_fe": spec["n_team_fe"],
            "n_opp_fe": spec["n_opp_fe"],
            "design_matrix_shape": spec["design_shape"],
            "best_alpha": spec["best_alpha"],
            "cv_per_alpha": {str(a): m for a, m in spec["per_alpha"].items()},
            "best_cv_rmse": spec["per_alpha"][spec["best_alpha"]]["rmse"],
            "best_cv_r2": spec["per_alpha"][spec["best_alpha"]]["r2"],
        }

    out = {
        "regression_m1_team_fe": _pack(spec_base),
        "regression_m2_team_and_opp_fe": _pack(spec_full),
        "baseline_team_mean": {"rmse": baseline_rmse, "r2": baseline_r2},
        "summary_m1": sum1,
        "summary_m2": sum2,
        "between_spec_correlation": {"pred_xg": float(rho), "rank": float(rho_rank)},
        "per_team": rows,
    }
    out_path = os.path.join(args.repo_root, args.output)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to: {out_path}")


if __name__ == "__main__":
    main()
