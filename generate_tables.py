import os
import json
from glob import glob

LATEX_TEMPLATE = """\\begin{{table*}}[t]
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{center}}
\\begin{{small}}
\\begin{{sc}}
\\begin{{tabular}}{{lccccccc}}
\\toprule
Model & WD~($\downarrow$) & DTW NN~($\downarrow$) & Spectral WD~($\downarrow$) & Temp Corr.~($\uparrow$) & Density~($\uparrow$) & Coverage~($\uparrow$) & Diversity \\\\
\\midrule
{rows}\\bottomrule
\\end{{tabular}}
\\end{{sc}}
\\end{{small}}
\\end{{center}}
\\vskip -0.1in
\\end{{table*}}
"""

def find_latest_top_k_dir(benchmark_path):
    top_k = [d for d in os.listdir(benchmark_path) if d.startswith('top_')]
    ks = [(int(d.split('_')[1]), d) for d in top_k if d.split('_')[1].isdigit()]
    return os.path.join(benchmark_path, max(ks, default=(None,None))[1]) if ks else None

from decimal import Decimal, getcontext, ROUND_HALF_UP

def format_metric_value(v):
    if not isinstance(v, float):
        return str(v)

    if v == 0.0:
        return "0.0000"

    getcontext().prec = 30
    sign = "-" if v < 0 else ""
    d = Decimal(str(abs(v)))

    # Values >= 1 → standard 2 decimals
    if d >= 1:
        return f"{v:.2f}"

    # Convert to fixed decimal representation
    s = format(d, "f")
    _, dec = s.split(".")

    # Find first non-zero digit
    idx = 0
    while idx < len(dec) and dec[idx] == "0":
        idx += 1

    if idx == len(dec):
        return "0.0000"

    # Keep first non-zero + two following digits
    decimals_to_keep = idx + 3

    quant = Decimal("1e-" + str(decimals_to_keep))
    rounded = Decimal(str(v)).quantize(quant, rounding=ROUND_HALF_UP)

    return f"{rounded:.{decimals_to_keep}f}"



def extract_averaged_metrics(json_path):
    with open(json_path) as f:
        avg = json.load(f).get('averaged_metrics', {})
    keys = ["wasserstein","dtw_nn","spectral_wasserstein","temporal_correlation",
            "distributional_coverage.density","distributional_coverage.coverage","diversity_icd"]
    return [format_metric_value(v) if isinstance(v, float) else str(v) for v in (avg.get(k,"") for k in keys)]

def bold_best_metrics(metrics):
    if not metrics: return []
    num = 7
    cols = list(zip(*(row for _,row in metrics)))
    # Convert all numeric entries to float for finding bests, leave empty/None as None
    vals = []
    for col in cols:
        vals_col = []
        for x in col:
            try:
                vals_col.append(float(x))
            except (ValueError, TypeError):
                vals_col.append(None)
        vals.append(vals_col)
    # min for cols 0,1,2 (first 3), max for 3,4,5 (up to 6th), nothing for 6
    best = [(min([v for v in col if v is not None]) if i<3 else max([v for v in col if v is not None]))
            if i<6 else None for i,col in enumerate(vals)]
    idx = [[j for j,v in enumerate(col) if v==b] if b is not None and i!=6 else [] for i,(col,b) in enumerate(zip(vals,best))]
    out = []
    for mi,(name,row) in enumerate(metrics):
        out.append((name, [f"\\textbf{{{cell}}}" if mi in idx[i] else cell for i,cell in enumerate(row)]))
    return out

def generate_table_tex(benchmark, metrics_by_model, k_val, model_order=None):
    models = model_order or sorted(m for m,_ in metrics_by_model)
    mm = dict(metrics_by_model)
    rows = bold_best_metrics([(m,mm.get(m,[""]*7)) for m in models])
    body = "".join([f"{m} & " + " & ".join(vals) + " \\\\\n" for m, vals in rows])
    return LATEX_TEMPLATE.format(
        caption=f"{benchmark.capitalize()} Evaluation Results",
        label=f"{benchmark}-result",
        rows=body
    )

def get_model_name_from_filename(f): return os.path.basename(f).rsplit('.',1)[0].split('_')[-1]

def main():
    root = "data/evaluation_results"
    output_dir = "generated_tables"
    os.makedirs(output_dir, exist_ok=True)
    for bn in filter(lambda x: os.path.isdir(os.path.join(root, x)), os.listdir(root)):
        latest = find_latest_top_k_dir(os.path.join(root, bn))
        if not latest:
            continue
        metrics = [
            (get_model_name_from_filename(jf), extract_averaged_metrics(jf))
            for jf in sorted(glob(os.path.join(latest, "*_retrieval_evaluation_*.json")))
        ]
        output_path = os.path.join(output_dir, f"results_{bn}.tex")
        with open(output_path, "w") as f:
            f.write(generate_table_tex(bn, metrics, 0, None))

main()
