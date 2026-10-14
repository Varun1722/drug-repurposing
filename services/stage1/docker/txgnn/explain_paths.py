"""Build explanation paths for drug-disease pairs from TxGNN's GraphMask edge scores.

This follows the TxGNN Explorer notebook (demo.ipynb in the bundle) without building the whole
8M-edge graph in memory:
  1. GraphMask gave every KG edge a score in each of the 2 GNN layers (layer1_att, layer2_att).
  2. For the disease and for the drug, keep the top K neighbours per relation type, then the
     top K2 neighbours of those (2 hops).
  3. Take the subgraph on those nodes and list every simple path disease -> drug of up to 4 edges.
  4. Score a path as the mean of (layer1 + layer2) over its edges; group by meta-path.

The notebook picks neighbours with heapq.nlargest on NEGATED weights, which selects the LOWEST
scoring neighbours. Both behaviours are run so the difference is visible ("top" = highest scores,
"notebook" = what the notebook code does).
"""

import argparse
import itertools
import os
import time

import networkx as nx
import numpy as np
import pandas as pd

NOT_COOL = {
    "rev_contraindication",
    "contraindication",
    "drug_drug",
    "rev_off-label use",
    "off-label use",
    "anatomy_protein_absent",
    "rev_anatomy_protein_absent",
}


def load_scores(path):
    d = pd.read_pickle(path)
    d = d[~d.y_name.str.contains("CYP") & ~d.x_name.str.contains("CYP")]
    d = d.rename(columns={"indication_layer1_att": "layer1_att", "indication_layer2_att": "layer2_att"})
    d = d[["x_name", "y_name", "x_type", "y_type", "relation", "layer1_att", "layer2_att"]].copy()
    d["w"] = d["layer1_att"] + d["layer2_att"]
    for col in ["x_name", "y_name", "x_type", "y_type", "relation"]:
        d[col] = d[col].astype("category")
    return d.reset_index(drop=True)


def pick_neighbours(df, rows_by_node, node, k, mode):
    rows = rows_by_node.get(node)
    if rows is None:
        return set(), 0.0
    sub = df.iloc[rows]
    chosen = []
    for _, g in sub.groupby("relation", observed=True):
        chosen.append(g.nlargest(k, "w") if mode == "top" else g.nsmallest(k, "w"))
    chosen = pd.concat(chosen)
    return set(chosen["y_name"].astype(str)), float(chosen["w"].mean())


def neighbourhood(df, rows_by_node, node, k, k2, mode):
    first, w1 = pick_neighbours(df, rows_by_node, node, k, mode)
    second, weights = set(), [w1]
    for n in first:
        got, w = pick_neighbours(df, rows_by_node, n, k2, mode)
        second |= got
        weights.append(w)
    return first | second, float(np.mean(weights))


def paths_for_pair(df, rows_by_node, disease, drug, mode, k=10, k2=10, max_paths=300000, min_layer=0.0, min_sum=0.0):
    nb_x, w_x = neighbourhood(df, rows_by_node, disease, k, k2, mode)
    nb_y, w_y = neighbourhood(df, rows_by_node, drug, k, k2, mode)
    nodes = nb_x | nb_y | {disease, drug}
    sub = df[df["x_name"].astype(str).isin(nodes) & df["y_name"].astype(str).isin(nodes)]
    # Optionally keep only edges whose gate is open (>= min_layer) in BOTH GNN layers.
    sub = sub[(sub["layer1_att"] >= min_layer) & (sub["layer2_att"] >= min_layer)]
    # Or keep edges whose summed gate (layer1 + layer2, range 0-2) is at least min_sum.
    sub = sub[sub["w"] >= min_sum]
    g = nx.MultiDiGraph()
    for r in sub.itertuples(index=False):
        g.add_node(str(r.x_name), node_type=str(r.x_type))
        g.add_node(str(r.y_name), node_type=str(r.y_type))
        g.add_edge(str(r.x_name), str(r.y_name), relation=str(r.relation), w=r.w)
    info = {"nodes": len(nodes), "edges": g.number_of_edges(), "mean_w_selected": (w_x + w_y) / 2}
    if disease not in g or drug not in g:
        return pd.DataFrame(), info
    rows, n_seen = [], 0
    gen = nx.all_simple_paths(g, source=disease, target=drug, cutoff=4)
    for p in itertools.islice(gen, max_paths):
        n_seen += 1
        rels, ws = [], []
        for a, b in zip(p[:-1], p[1:]):
            e = g[a][b][0]
            rels.append(e["relation"])
            ws.append(e["w"])
        if NOT_COOL.intersection(rels):
            continue
        meta = []
        for n, r in zip(p[:-1], rels):
            meta += [g.nodes[n]["node_type"], r]
        meta.append(g.nodes[p[-1]]["node_type"])
        text = [p[0]]
        for r, n in zip(rels, p[1:]):
            text += [r[4:] if r.startswith("rev_") else r, n]
        rows.append(
            {
                "meta_path": " -> ".join(m[4:] if m.startswith("rev_") else m for m in meta),
                "path": " -> ".join(text),
                "n_edges": len(rels),
                "score": float(np.mean(ws)),
            }
        )
    info["paths_seen"] = n_seen
    info["truncated"] = n_seen >= max_paths
    return pd.DataFrame(rows), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="/weights/model_ckpt/graphmask_output_indication.pkl")
    ap.add_argument("--disease", required=True)
    ap.add_argument("--drugs", nargs="+", required=True)
    ap.add_argument("--out", default="/out")
    ap.add_argument("--min-layer", type=float, default=0.0)
    ap.add_argument("--min-sum", type=float, default=0.0)
    ap.add_argument("--modes", nargs="+", default=["top", "notebook"])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    t0 = time.time()
    df = load_scores(args.scores)
    print(f"[{time.time() - t0:5.0f}s] loaded {len(df):,} scored edges; "
          f"{df.memory_usage(deep=True).sum() / 1e9:.1f} GB", flush=True)
    q = df["w"].quantile([0.5, 0.9, 0.99]).round(4).to_dict()
    print(f"edge score (layer1+layer2, range 0-2): quantiles {q}; mean {df['w'].mean():.4f}; "
          f"share above 1.0: {(df['w'] > 1.0).mean():.4f}", flush=True)
    both = ((df["layer1_att"] >= 0.5) & (df["layer2_att"] >= 0.5)).mean()
    print(f"edges open (>= 0.5) in both layers: {both:.4f}; layer1 >= 0.5: "
          f"{(df['layer1_att'] >= 0.5).mean():.4f}; layer2 >= 0.5: {(df['layer2_att'] >= 0.5).mean():.4f}",
          flush=True)
    rows_by_node = df.groupby("x_name", observed=True).indices
    for node in [args.disease, *args.drugs]:
        t = df.loc[df["x_name"] == node, "x_type"]
        print(f"node {node!r}: {len(rows_by_node.get(node, []))} outgoing edges, type "
              f"{t.iloc[0] if len(t) else None}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    for drug in args.drugs:
        for mode in args.modes:
            t1 = time.time()
            paths, info = paths_for_pair(
                df, rows_by_node, args.disease, drug, mode, min_layer=args.min_layer, min_sum=args.min_sum
            )
            print(f"\n=== {args.disease} <- {drug} | neighbour pick: {mode} | "
                  f"{info['nodes']} nodes, {info['edges']} edges, {len(paths)} usable paths "
                  f"({'capped' if info.get('truncated') else 'complete'}), "
                  f"mean edge score of picked neighbours {info['mean_w_selected']:.3f} "
                  f"({time.time() - t1:.0f}s)", flush=True)
            if paths.empty:
                continue
            paths.to_csv(os.path.join(args.out, f"paths_{args.disease}_{drug}_{mode}{args.tag}.csv".replace(" ", "_")), index=False)
            meta = (paths.groupby("meta_path").agg(n=("score", "size"), mean=("score", "mean"), best=("score", "max"))
                    .sort_values("best", ascending=False))
            print("top meta-paths (by best path):")
            print(meta.head(6).round(3).to_string(), flush=True)
            print("top paths:")
            for r in paths.sort_values("score", ascending=False).head(8).itertuples():
                print(f"  {r.score:.3f}  {r.path}", flush=True)


if __name__ == "__main__":
    main()
