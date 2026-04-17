"""Generate LaTeX tables from xg_validation_results.json for inclusion in main.tex.

Outputs:
  paper/table_validation.tex  -- xG validation per-team summary
  paper/table_optimal_xi.tex  -- optimal XI for four representative teams
"""
import json
import os

TEAM_DISPLAY = {
    "1. FC Heidenheim 1846":       "Heidenheim",
    "1. FC Köln":                  "Köln",
    "1. FC Union Berlin":          "Union Berlin",
    "1. FSV Mainz 05":             "Mainz 05",
    "Bayer 04 Leverkusen":         "Leverkusen",
    "Borussia Dortmund":           "Dortmund",
    "Borussia Mönchengladbach":    "M'gladbach",
    "Eintracht Frankfurt":         "Frankfurt",
    "FC Augsburg":                 "Augsburg",
    "FC Bayern München":           "Bayern",
    "RasenBallsport Leipzig":      "Leipzig",
    "SC Freiburg":                 "Freiburg",
    "SV Darmstadt 98":             "Darmstadt",
    "SV Werder Bremen":            "W.\\ Bremen",
    "TSG 1899 Hoffenheim":         "Hoffenheim",
    "VfB Stuttgart":               "Stuttgart",
    "VfL Bochum":                  "Bochum",
    "VfL Wolfsburg":               "Wolfsburg",
}

# 2023/24 final league positions (Bundesliga)
LEAGUE_POSITION = {
    "Bayer 04 Leverkusen":        1,
    "VfB Stuttgart":              2,
    "FC Bayern München":          3,
    "RasenBallsport Leipzig":     4,
    "Borussia Dortmund":          5,
    "Eintracht Frankfurt":        6,
    "TSG 1899 Hoffenheim":        7,
    "1. FC Heidenheim 1846":      8,
    "SV Werder Bremen":           9,
    "SC Freiburg":                10,
    "FC Augsburg":                11,
    "VfL Wolfsburg":              12,
    "1. FSV Mainz 05":            13,
    "Borussia Mönchengladbach":   14,
    "1. FC Union Berlin":         15,
    "VfL Bochum":                 16,
    "1. FC Köln":                 17,
    "SV Darmstadt 98":            18,
}


def safe_latex(s: str) -> str:
    return (s.replace("&", "\\&").replace("%", "\\%"))


def build_validation_table(results: dict) -> str:
    rows = sorted(
        results["per_team"],
        key=lambda r: LEAGUE_POSITION.get(r["team_name"], 99),
    )
    # Regression metadata for caption
    r1 = results["regression_m1_team_fe"]
    r2 = results["regression_m2_team_and_opp_fe"]

    caption = (
        "External validation of network-optimized lineups against observed match-day "
        "selections under two Ridge regressions: \\textbf{M1} (players + team FE, "
        f"$R^2 = {r1['best_cv_r2']:.3f}$) and \\textbf{{M2}} (players + team FE + opponent FE, "
        f"$R^2 = {r2['best_cv_r2']:.3f}$). "
        "For each team we report the predicted xG of the optimized XI "
        "($\\widehat{\\mathrm{xG}}_{\\mathrm{opt}}$), the mean predicted xG across the "
        "team's 34 observed XIs ($\\widehat{\\mathrm{xG}}_{\\mathrm{obs}}$), the uplift "
        "$\\Delta = \\widehat{\\mathrm{xG}}_{\\mathrm{opt}} - \\widehat{\\mathrm{xG}}_{\\mathrm{obs}}$, "
        "and the rank of the optimized XI among the 34 observed XIs (higher is better). "
        "Predictions are made against the baseline opponent for a fair comparison. "
        "Teams are ordered by 2023/24 final league position."
    )

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        f"\\caption{{{caption}}}",
        "\\label{tab:xg_validation}",
        "\\begin{tabular}{@{}l l r r r r r r r r@{}}",
        "\\toprule",
        "& & \\multicolumn{4}{c}{\\textbf{M1: team FE only}} & \\multicolumn{4}{c}{\\textbf{M2: team FE + opponent FE}} \\\\",
        "\\cmidrule(lr){3-6} \\cmidrule(lr){7-10}",
        "Team & Form. & "
        "$\\widehat{\\mathrm{xG}}_{\\mathrm{opt}}$ & $\\widehat{\\mathrm{xG}}_{\\mathrm{obs}}$ & $\\Delta$ & Rank & "
        "$\\widehat{\\mathrm{xG}}_{\\mathrm{opt}}$ & $\\widehat{\\mathrm{xG}}_{\\mathrm{obs}}$ & $\\Delta$ & Rank \\\\",
        "\\midrule",
    ]
    for r in rows:
        d1 = r["m1_pred_xg_optimized"] - r["m1_pred_xg_obs_mean"]
        d2 = r["m2_pred_xg_optimized"] - r["m2_pred_xg_obs_mean"]
        lines.append(
            "{team} & {fmt} & "
            "{p1:.2f} & {o1:.2f} & {d1:+.2f} & {r1}/{n} & "
            "{p2:.2f} & {o2:.2f} & {d2:+.2f} & {r2}/{n} \\\\".format(
                team=TEAM_DISPLAY[r["team_name"]],
                fmt=r["formation"],
                p1=r["m1_pred_xg_optimized"],
                o1=r["m1_pred_xg_obs_mean"],
                d1=d1,
                r1=r["m1_rank_among_observed_xis"],
                p2=r["m2_pred_xg_optimized"],
                o2=r["m2_pred_xg_obs_mean"],
                d2=d2,
                r2=r["m2_rank_among_observed_xis"],
                n=r["n_observed_xis"],
            )
        )
    n = len(rows)
    mean_d1 = sum(r["m1_pred_xg_optimized"] - r["m1_pred_xg_obs_mean"] for r in rows) / n
    mean_d2 = sum(r["m2_pred_xg_optimized"] - r["m2_pred_xg_obs_mean"] for r in rows) / n
    mean_rank1 = sum(r["m1_rank_among_observed_xis"] for r in rows) / n
    mean_rank2 = sum(r["m2_rank_among_observed_xis"] for r in rows) / n
    lines.append("\\midrule")
    lines.append(
        "\\textbf{{Mean}} & --- & "
        "--- & --- & \\textbf{{{d1:+.2f}}} & \\textbf{{{r1:.1f}/34}} & "
        "--- & --- & \\textbf{{{d2:+.2f}}} & \\textbf{{{r2:.1f}/34}} \\\\".format(
            d1=mean_d1, r1=mean_rank1, d2=mean_d2, r2=mean_rank2
        )
    )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def build_optimal_xi_table(results: dict, teams_to_include: list) -> str:
    per = {r["team_name"]: r for r in results["per_team"]}
    rows = []
    for t in teams_to_include:
        r = per[t]
        formation = r["formation"]
        slots = None
        for fmt_name, slot_list in FORMATION_SLOTS.items():
            if fmt_name == formation:
                slots = slot_list
                break
        xi = r["optimal_xi"]
        rows.append((TEAM_DISPLAY[t], formation, slots, xi))

    # Print as a 2-column table per team
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Network-optimized starting XIs for four representative teams under default "
        "objective weights ($\\alpha=0.55$, $\\beta=0.25$, $w_{\\mathrm{coh}}=0.20$). "
        "Formation is chosen as the highest-preference template feasible given the squad's "
        "positional eligibility.}",
        "\\label{tab:optimal_xi}",
        "\\begin{tabular}{@{}l l l@{}}",
        "\\toprule",
        "Team (Formation) & Slot & Player \\\\",
        "\\midrule",
    ]
    for i, (name, fmt, slots, xi) in enumerate(rows):
        header = f"\\multirow{{11}}{{*}}{{\\textbf{{{name}}} ({fmt})}}"
        for j, (slot, player) in enumerate(zip(slots, xi)):
            left = header if j == 0 else ""
            player_esc = safe_latex(player)
            lines.append(f"{left} & {slot} & {player_esc} \\\\")
        if i < len(rows) - 1:
            lines.append("\\midrule")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


# Mirror of FORMATION_TEMPLATES in optimization.py -- keep in sync
FORMATION_SLOTS = {
    "4-3-3":   ["GK","LB","CB1","CB2","RB","DM","CM","AM","LW","ST","RW"],
    "4-2-3-1": ["GK","LB","CB1","CB2","RB","DM1","DM2","LW","AM","RW","ST"],
    "4-4-2":   ["GK","LB","CB1","CB2","RB","LM","CM1","CM2","RM","ST1","ST2"],
    "3-4-3":   ["GK","CB1","CB2","CB3","LWB","CM1","CM2","RWB","LW","ST","RW"],
    "3-5-2":   ["GK","CB1","CB2","CB3","LWB","DM","CM1","CM2","RWB","ST1","ST2"],
    "5-3-2":   ["GK","LWB","CB1","CB2","CB3","RWB","CM1","CM2","CM3","ST1","ST2"],
    "4-1-4-1": ["GK","LB","CB1","CB2","RB","DM","LM","CM1","CM2","RM","ST"],
}


def main():
    with open("xg_validation_results.json") as f:
        results = json.load(f)

    os.makedirs("paper", exist_ok=True)

    # Validation table
    val_tex = build_validation_table(results)
    with open("paper/table_validation.tex", "w") as f:
        f.write(val_tex + "\n")
    print(f"Wrote paper/table_validation.tex  ({len(val_tex)} bytes)")

    # Optimal-XI table for four representative teams (top, middle, bottom)
    representative = [
        "FC Bayern München",
        "Bayer 04 Leverkusen",
        "Eintracht Frankfurt",
        "SV Darmstadt 98",
    ]
    xi_tex = build_optimal_xi_table(results, representative)
    with open("paper/table_optimal_xi.tex", "w") as f:
        f.write(xi_tex + "\n")
    print(f"Wrote paper/table_optimal_xi.tex  ({len(xi_tex)} bytes)")


if __name__ == "__main__":
    main()
