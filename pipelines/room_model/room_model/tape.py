"""Tape measurements against the scan and LLM values, as a French Markdown report."""
from __future__ import annotations

from room_model.fmt import fr

NONE = "–"


def signed_cm(delta_m: float) -> str:
    cm = round(delta_m * 100, 1)
    return f"{'-' if cm < 0 else '+'}{fr(abs(cm))}"


def compare(arch: dict, tape: dict) -> str:
    """Per measured dimension: scan value, LLM value (when corrected), tape value and the tape's gap to each, in cm.

    `tape` is {"<dimension id>": <metres>}; ids unknown to the architecture are listed apart.
    """
    dims = {d["id"]: d for d in arch["dimensions"]}
    rows, gaps, unknown = [], [], []
    for dim_id, measured in tape.items():
        dim = dims.get(dim_id)
        if dim is None:
            unknown.append(dim_id)
            continue
        corrected = "scan_value" in dim
        scan, sigma = (dim["scan_value"], dim.get("scan_sigma", dim["sigma"])) if corrected else (dim["value"], dim["sigma"])
        gap = measured - scan
        gaps.append((abs(gap), dim_id, gap))
        llm = fr(dim["value"] * 100) if corrected else NONE
        llm_gap = signed_cm(measured - dim["value"]) if corrected else NONE
        verdict = "dans l'incertitude du scan" if abs(gap) <= 2 * sigma else "hors de l'incertitude du scan"
        rows.append(f"| {dim_id} | {dim['label']} | {fr(scan * 100)} | {llm} | {fr(measured * 100)} | {signed_cm(gap)} | {llm_gap} | {verdict} |")
    out = ["# Comparaison au mètre ruban", "",
           "| Cote | Description | Scan (cm) | LLM (cm) | Mètre (cm) | Écart mètre − scan (cm) | Écart mètre − LLM (cm) | Verdict |",
           "|---|---|---|---|---|---|---|---|", *rows, ""]
    if gaps:
        mean = sum(g[0] for g in gaps) / len(gaps)
        worst = max(gaps)
        out += [f"{len(gaps)} cote(s) comparée(s) ; écart moyen au scan {fr(mean * 100)} cm ; "
                f"plus grand écart : {worst[1]} ({signed_cm(worst[2])} cm).", ""]
    if unknown:
        out += [f"Cotes inconnues de l'architecture, ignorées : {', '.join(unknown)}.", ""]
    return "\n".join(out)
