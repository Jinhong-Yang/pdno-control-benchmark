"""Generate the figure guide from verified current figure provenance."""
from pathlib import Path
import hashlib
import json

PLANNED = {
    "figD06_candidate_design": "Matched candidate-design diagnostic (D6)",
    "figD07_error_decomposition": "Observer and operator error diagnostic (D7)",
}
ORIGINAL = {"fig02_prediction", "fig03_control", "fig04_latency",
            "fig05_h1", "fig06_heat", "figS01_training"}
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def build(root):
    root = Path(root)
    records = json.loads((root/"data/figure_provenance.json").read_text())
    assert len({r["id"] for r in records}) == len(records)
    lines = [
        "# Figure design and data map", "",
        "This guide is regenerated from `data/figure_provenance.json`. Asset identifiers",
        "are file names, not final manuscript figure numbers. The manuscript and supplement",
        "determine the in-paper numbering and placement.", "",
        f"The current provenance contains **{len(records)} generated figures**. The guide builder",
        "checks each recorded input hash and figure PDF hash before listing it. This",
        "check establishes file correspondence; final-page visual review and scientific",
        "claim validation are separate requirements.", "",
        "Run `python scripts/build_figures.py` from this publication directory to regenerate",
        "all figures whose required inputs are complete, followed by this guide. No model",
        "training or GPU timing is launched by figure generation. Figure 1 uses editable",
        "TikZ source in `figures/fig01_design.tex`; regeneration requires pdfLaTeX, TikZ",
        "and standalone. The supplied PDFs are the primary LaTeX assets; SVG and PNG",
        "copies support editing and previews.", "",
        "| Asset | Evidence population / scope | Description | Input files |",
        "|---|---|---|---|",
    ]
    def clean(value):
        return str(value).replace("|", r"\|").replace("\n", " ")
    checks = []
    for r in records:
        for item in r["input_sha256"]:
            p = root/Path(item["path"].replace("\\", "/"))
            assert p.is_file() and sha(p)==item["sha256"], ("stale figure input", r["id"], str(p))
            checks.append(str(p.relative_to(root)))
        pdf = root/"figures"/(r["id"]+".pdf")
        assert pdf.is_file() and sha(pdf)==r["pdf_sha256"], ("stale figure PDF", r["id"])
        assert all((root/"figures"/(r["id"]+"."+ext)).is_file() for ext in ("svg","png"))
        scope = ("Primary evaluation" if r["id"] in ORIGINAL else
                 "Implemented design and follow-up paths" if r["id"]=="fig01_design" else
                 "Follow-up diagnostic; populations distinguished explicitly")
        sources = ", ".join("`"+x["path"].replace("\\","/")+"`" for x in r["input_sha256"])
        description = r["description"]
        for old, new in [("original/revision", "primary/follow-up"), ("original", "primary"), ("Original", "Primary"), ("Revision", "Follow-up"), ("revision", "follow-up"), ("oracles", "reference-solver controls"), ("oracle", "reference-solver"), ("parents", "scenarios"), ("parent", "scenario"), ("E3", "D3"), ("E5", "D5")]:
            description = description.replace(old, new)
        lines.append("| "+ " | ".join(map(clean, (r["id"],scope,description,sources)))+" |")
    missing = {name: detail for name, detail in PLANNED.items()
               if name not in {r["id"] for r in records}}
    if missing:
        lines.extend(["", "The following planned assets are **not yet generated** because their",
                      "complete experimental inputs remain pending:", ""])
        lines.extend("- `"+name+"`: "+detail+"." for name,detail in missing.items())
        lines.extend(["", "Their absence is not a zero effect or evidence that an experiment completed."])
    lines.extend([
        "", "P is orange, B4 is blue, B0 is dark gray, and other controls use neutral",
        "colors. Seed markers and line styles supplement color. Scientific captions",
        "must distinguish fixed-denominator cost intervals, joint-denominator sensitivity",
        "intervals, hierarchical seed/session latency intervals, and descriptive pooled",
        "quantiles. These are different statistics and must not be interchanged.", "",
        "The signed-heat event figure (`fig06_heat`) belongs to Supplement S7.",
        "The unused all-zero NH constraint asset is retained for source continuity; it is not included in the manuscript.",
        "SH reference-solver controls must not be plotted as if evaluated on the NH population.",
        "The main control figure uses a symmetric logarithmic axis outside the central",
        "linear region and must retain that explanation in its caption.", "",
        "Figure 1 depicts implemented information flow, including the diagnostic reference-solver",
        "and cache paths. It is not a physical apparatus or a simulated field image.",
        "Long protocol qualifications belong in Supplement S2 and the Discussion;",
        "sampling units, axes, interval definitions and population distinctions remain",
        "in the relevant captions.", "",
    ])
    (root/"FIGURE_GUIDE.md").write_text("\n".join(lines),encoding="utf-8")
    receipt = {"status":"PASS_CURRENT_FIGURE_FILE_CORRESPONDENCE",
               "figure_count":len(records), "pending_assets":list(missing),
               "provenance_sha256":sha(root/"data/figure_provenance.json"),
               "guide_sha256":sha(root/"FIGURE_GUIDE.md"),
               "input_checks":checks,
               "limits":"Hash correspondence and asset existence only, not visual or scientific validation."}
    (root/"data/FIGURE_GUIDE_CHECK.json").write_text(json.dumps(receipt,indent=2)+"\n")
    return receipt

if __name__ == "__main__":
    result=build(Path(__file__).resolve().parents[1])
    print(json.dumps({k:result[k] for k in ("status","figure_count","pending_assets")}))
