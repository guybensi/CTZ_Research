"""Decision rule: explore, then focus (PLAN.md section 6).

Applied to the benchmark summary (benchmark.summarize). In order:

1. Gates. A method is dropped if
   * its physics checks fail (far-field pure clear sky outside 80 +- 10 W m^-2, values above
     S0 mu0, a power-spectrum peak at the footprint spacing);
   * its residuals are not flat against SZA, VZA, RAA or glint (trend effect > trend_gate and
     p < 0.01), or the swath-side (forward vs backscatter) asymmetry exceeds swath_gate;
   * its ranking flips between the primary operator and a sensitivity operator (uniform /
     Gaussian PSF): against some other map method, the sign of the macro-RMSE difference changes
     and both differences exceed flip_tol.
2. Primary score: macro-averaged RMSE over the clear / broken / overcast regimes.
3. Walk the passing methods from simple to complex. A more complex method replaces the incumbent
   only if (a) the paired bootstrap CI of the unit-loss difference excludes 0 in its favour after
   Holm correction across pairs, and (b) its macro-RMSE gain exceeds the CERES noise floor of the
   regimes it was scored on (mean of noise_floor_by_regime over those regimes).
4. Ties (no significant difference with the winner) are broken by, in order: synthetic-truth pixel
   RMSE (a > 5% relative margin is needed), a smaller gap between the interleave and LOGO scores
   (a > 0.5 W m^-2 margin), then the default: M3, or M2 if image data is scarce, then simplicity.
5. The UNet (M5) must beat the hybrid M6b, not only M3: if M5 wins but is not significantly better
   than M6b, M6b wins, and the log says that spatial context adds little.
Methods that read test labels (B-const, M6e, M7, M9-local) are never ranked.
"""
from __future__ import annotations

import numpy as np

REGIMES = ("clear", "broken", "overcast")
PIXEL_MARGIN = 0.05      # tie-breaker 1: relative pixel-RMSE margin
INTERLEAVE_MARGIN = 0.5  # tie-breaker 2: W m^-2


def _get(cfg, key, default):
    return getattr(cfg, key, default)


def regime_noise_floor(res: dict, cfg) -> float:
    """Noise floor a method must beat: mean regime floor over the regimes it was scored on.

    Example:
        >>> class C: noise_floor = 2.0; noise_floor_by_regime = {"clear": 2.0, "broken": 10.0}
        >>> regime_noise_floor({"by_regime": {"clear": {"n": 5}, "broken": {"n": 5}}}, C())
        6.0
        >>> regime_noise_floor({}, C())
        2.0
    """
    floors = _get(cfg, "noise_floor_by_regime", None) or {}
    by = res.get("by_regime") or {}
    vals = [floors[r] for r in REGIMES if r in by and by[r].get("n", 0) > 0 and r in floors]
    return float(np.mean(vals)) if vals else float(_get(cfg, "noise_floor", 2.0))


def ranking_flips(summary: dict, names: list[str], operators: tuple[str, ...], tol: float = 0.5) -> dict[str, list[str]]:
    """Pairs of map methods whose order flips between the primary macro RMSE and a sensitivity operator's.

    Footprint-only methods (produces_map False: M0, M1) predict one value per footprint whatever the
    operator, so they take no part in this check.

    Example:
        >>> s = {"models": {"A": {"macro_rmse": 5.0, "macro_rmse_uniform": 9.0},
        ...                 "B": {"macro_rmse": 7.0, "macro_rmse_uniform": 6.0}}}
        >>> ranking_flips(s, ["A", "B"], ("uniform",))
        {'A': ['B@uniform'], 'B': ['A@uniform']}

    Returns:
        {method: ["other@operator", ...]} for every method involved in a flip.
    """
    out: dict[str, list[str]] = {}
    m = summary["models"]
    names = [n for n in names if m[n].get("produces_map", True)]
    for op in operators:
        key = f"macro_rmse_{op}"
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                pa, pb = m[a].get("macro_rmse", np.nan), m[b].get("macro_rmse", np.nan)
                sa, sb = m[a].get(key, np.nan), m[b].get(key, np.nan)
                if not np.all(np.isfinite([pa, pb, sa, sb])):
                    continue
                if (pa - pb) * (sa - sb) < 0 and abs(pa - pb) > tol and abs(sa - sb) > tol:
                    out.setdefault(a, []).append(f"{b}@{op}")
                    out.setdefault(b, []).append(f"{a}@{op}")
    return out


def gate_failures(name: str, res: dict, cfg, flips: dict) -> list[str]:
    """List of the gates (section 6, rule 1) that a method fails; empty if it passes."""
    fails = []
    phys = res.get("physics")
    if phys is not None and not phys.get("passed", True):
        fails += [k for k in ("ok_pcs", "ok_bounds", "ok_spacing") if not phys.get(k, True)] or ["physics"]
    for c, t in res.get("trend_effect", {}).items():
        if t["effect_wm2"] > _get(cfg, "trend_gate", 5.0) and t["p"] < 0.01:
            fails.append(f"trend:{c}")
    if abs(res.get("swath_asymmetry", 0.0) or 0.0) > _get(cfg, "swath_gate", 5.0):
        fails.append("swath_asymmetry")
    if name in flips:
        fails.append(f"ranking_flip:{','.join(flips[name])}")
    return fails


def compare(summary: dict, cand: str, inc: str) -> tuple[bool, bool, dict | None]:
    """(cand significantly better, inc significantly better, comparison dict) from the paired tests.

    Uses the Holm-adjusted bootstrap p-value and the bootstrap CI when present, else the
    Holm-adjusted Wilcoxon p-value with the sign of the RMSE difference.
    """
    comps = summary.get("comparisons", {})
    comp, sign = comps.get(f"{cand} vs {inc}"), 1.0
    if comp is None:
        comp, sign = comps.get(f"{inc} vs {cand}"), -1.0
    if comp is None:
        return False, False, None
    if "p_holm_boot" in comp and "ci_low" in comp:
        sig = comp["p_holm_boot"] < 0.05
        lo, hi = sign * comp["ci_low"], sign * comp["ci_high"]   # CI of loss(cand) - loss(inc)
        lo, hi = min(lo, hi), max(lo, hi)
        return sig and hi < 0, sig and lo > 0, comp
    sig = comp.get("p_holm", 1.0) < 0.05
    m = summary["models"]
    diff = m[cand].get("macro_rmse", np.nan) - m[inc].get("macro_rmse", np.nan)
    return sig and diff < 0, sig and diff > 0, comp


def _scarce(summary: dict, cfg) -> bool:
    flag = _get(cfg, "image_data_scarce", None)
    if flag is not None:
        return bool(flag)
    return summary.get("n_image_granules", 0) < 10


def break_tie(summary: dict, tie: list[str], cfg) -> tuple[str, str]:
    """Pick one method from a tie set; returns (winner, reason)."""
    m = summary["models"]
    px = {n: m[n].get("pixel", {}).get("rmse", np.nan) for n in tie}
    if all(np.isfinite(v) for v in px.values()):
        order = sorted(tie, key=lambda n: px[n])
        if px[order[1]] - px[order[0]] > PIXEL_MARGIN * px[order[1]]:
            return order[0], f"tie-breaker 1 (synthetic pixel RMSE {px[order[0]]:.2f} vs {px[order[1]]:.2f})"
    gap = {n: abs(m[n].get("interleave", {}).get("rmse", np.nan) - m[n].get("overall", {}).get("rmse", np.nan)) for n in tie}
    if all(np.isfinite(v) for v in gap.values()):
        order = sorted(tie, key=lambda n: gap[n])
        if gap[order[1]] - gap[order[0]] > INTERLEAVE_MARGIN:
            return order[0], f"tie-breaker 2 (interleave-LOGO gap {gap[order[0]]:.2f} vs {gap[order[1]]:.2f})"
    default = "M2" if _scarce(summary, cfg) else "M3"
    if default in tie:
        return default, f"default on a tie ({default}{', image data scarce' if default == 'M2' else ''})"
    simplest = min(tie, key=lambda n: (m[n].get("complexity", 0), m[n].get("macro_rmse", np.inf)))
    return simplest, "tie-breaker 3 (simplicity)"


def decide(summary: dict, cfg) -> dict:
    """Apply the section 6 rule; returns winner, ranking, tie set, gate failures and a log.

    Example:
        >>> s = {"models": {"A": {"overall": {"rmse": 9}, "macro_rmse": 9.0, "complexity": 0}},
        ...      "comparisons": {}}
        >>> decide(s, None)["winner"]
        'A'
    """
    log, passing, gates = [], [], {}
    models = summary["models"]
    ranked = [n for n, r in models.items() if not r.get("uses_test_labels") and r.get("overall")
              and np.isfinite(r.get("macro_rmse", np.nan))]
    for n, r in models.items():
        if r.get("uses_test_labels"):
            log.append(f"{n} reads test labels: reported, never ranked")
    flips = ranking_flips(summary, ranked, tuple(_get(cfg, "sensitivity_operators", ())), _get(cfg, "flip_tol", 0.5))
    for name in ranked:
        fails = gate_failures(name, models[name], cfg, flips)
        gates[name] = fails
        if fails:
            log.append(f"{name} fails gates: {fails}")
        else:
            passing.append(name)
    if not passing:
        return {"winner": None, "ranking_by_macro_rmse": [], "tie_set": [], "gates": gates,
                "log": log + ["no model passed the gates"]}
    passing.sort(key=lambda n: (models[n].get("complexity", 0), models[n]["macro_rmse"]))
    incumbent = passing[0]
    for cand in passing[1:]:
        better, worse, comp = compare(summary, cand, incumbent)
        gain = models[incumbent]["macro_rmse"] - models[cand]["macro_rmse"]
        floor = regime_noise_floor(models[cand], cfg)
        if better and gain > floor:
            log.append(f"{cand} replaces {incumbent}: macro RMSE gain {gain:.2f} > noise floor {floor:.2f} W/m2, "
                       f"significant after Holm")
            incumbent = cand
        else:
            why = "not significant" if not better else f"gain {gain:.2f} <= noise floor {floor:.2f} W/m2"
            log.append(f"{cand} does not replace {incumbent}: gain {gain:.2f} W/m2 ({why})")
    # tie: every passing method the winner does not significantly beat (a significant gain below the noise floor is a tie)
    tie = [incumbent] + [n for n in passing if n != incumbent and not compare(summary, incumbent, n)[0]]
    winner = incumbent
    if len(tie) > 1:
        winner, reason = break_tie(summary, tie, cfg)
        log.append(f"tie between {tie}: {winner} chosen by {reason}")
    if winner == "M5" and "M6b" in passing:
        better, _, _ = compare(summary, "M5", "M6b")
        if not better:
            log.append("M5 (UNet) is not significantly better than M6b: M6b wins; spatial context adds little "
                       "beyond a residual correction (a result about the scale of the CTZ)")
            winner = "M6b"
    ranking = sorted(passing, key=lambda n: models[n]["macro_rmse"])
    return {"winner": winner, "ranking_by_macro_rmse": ranking, "tie_set": tie, "gates": gates, "log": log}
