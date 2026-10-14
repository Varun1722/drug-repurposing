"""T3 smoke test: load TxGNN's example weights on CPU and score every drug for some diseases.

Runs inside the heisenberg-txgnn image (Python 3.8). It answers four questions:
  1. Does the model load and run on CPU, and how long and how much memory does it take?
  2. Does scoring in one batch give the same numbers as scoring one disease at a time?
  3. Do the known indications of the test diseases rank high?
  4. Which diseases did the example weights see during training? (compared to the split we rebuild)
"""

import argparse
import os
import resource
import time

import numpy as np
import pandas as pd
import torch

INDICATION = ("drug", "indication", "disease")
CONTRA = ("drug", "contraindication", "disease")


def txgnn_key(node_id):
    """TxGNN's internal id string (its convert2str): plain numbers become floats, e.g. 7739.0."""
    if "_" in node_id:
        return node_id
    try:
        return str(float(node_id))
    except ValueError:
        return node_id


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB on Linux


def stamp(label, t0):
    print(f"[{time.time() - t0:7.1f}s | peak {peak_mb():7.0f} MB] {label}", flush=True)


def score(model, drug_idx, disease_idxs):
    """Return {disease_idx: DataFrame(drug_idx, indication, contraindication)}."""
    frames = []
    for d in disease_idxs:
        for rel in ("indication", "contraindication"):
            frames.append(pd.DataFrame({"x_idx": drug_idx, "relation": rel, "y_idx": d}))
    df = pd.concat(frames, ignore_index=True)
    with torch.no_grad():
        out = model.predict(df)
    result = {}
    ind = torch.sigmoid(out[INDICATION]).reshape(-1).numpy()
    con = torch.sigmoid(out[CONTRA]).reshape(-1).numpy()
    n = len(drug_idx)
    for i, d in enumerate(disease_idxs):
        result[d] = pd.DataFrame(
            {
                "drug_idx": drug_idx,
                "indication": ind[i * n : (i + 1) * n],
                "contraindication": con[i * n : (i + 1) * n],
            }
        )
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/primekg")
    ap.add_argument("--weights", default="/weights/model_ckpt")
    ap.add_argument("--diseases", default="/config/test_diseases.csv")
    ap.add_argument("--only", nargs="*", help="MONDO ids to score; default is every mapped one")
    ap.add_argument("--split", default="complex_disease")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="/out")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    if args.threads:
        torch.set_num_threads(args.threads)
    print(f"torch threads: {torch.get_num_threads()}", flush=True)
    from txgnn import TxData, TxGNN

    data = TxData(data_folder_path=args.data)
    data.prepare_split(split=args.split, seed=args.seed)
    stamp(f"split prepared ({args.split}, seed {args.seed})", t0)

    model = TxGNN(data=data, weight_bias_track=False, proj_name="x", exp_name="x", device="cpu")
    model.load_pretrained(args.weights)
    stamp(f"weights loaded; config = {model.config}", t0)

    ids = data.retrieve_id_mapping()
    disease_by_node = {v: k for k, v in ids["idx2id_disease"].items()}
    drug_idx = np.array(sorted(ids["idx2id_drug"]), dtype=np.int64)
    drug_name = {i: ids["id2name_drug"].get(ids["idx2id_drug"][i], "?") for i in drug_idx}
    print(f"drugs: {len(drug_idx)}, diseases in graph: {len(disease_by_node)}", flush=True)

    table = pd.read_csv(args.diseases, dtype={"txgnn_node_id": str}).dropna(
        subset=["txgnn_node_id"]
    )
    if args.only:
        table = table[table["mondo_id"].isin(args.only)]
    table["disease_idx"] = table["txgnn_node_id"].map(txgnn_key).map(disease_by_node)
    missing = table[table["disease_idx"].isna()]
    if len(missing):
        print("not found in TxGNN graph:", missing["name"].tolist(), flush=True)
    table = table.dropna(subset=["disease_idx"])
    table["disease_idx"] = table["disease_idx"].astype(int)

    df = data.df
    known = df[(df.relation == "indication") & (df.x_type == "drug")]

    def split_of(frame, d):
        f = frame[(frame.relation == "indication") & (frame.y_idx == d)]
        return set(f.x_idx)

    train, valid, test = data.df_train, data.df_valid, data.df_test

    first = int(table["disease_idx"].iloc[0])
    t1 = time.time()
    single = score(model, drug_idx, [first])[first]
    stamp(f"scored 1 disease x {len(drug_idx)} drugs in {time.time() - t1:.1f}s", t0)

    all_idx = [int(d) for d in table["disease_idx"]]
    t2 = time.time()
    batch = score(model, drug_idx, all_idx)
    stamp(f"scored {len(all_idx)} diseases in one batch in {time.time() - t2:.1f}s", t0)
    diff = np.abs(single["indication"].values - batch[first]["indication"].values).max()
    print(f"max |single - batched| indication score for the first disease: {diff:.2e}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    rows = []
    for row in table.itertuples():
        d = row.disease_idx
        s = batch[d].copy()
        s["rank"] = s["indication"].rank(ascending=False, method="first").astype(int)
        k_all = split_of(known, d)
        k_train, k_valid, k_test = split_of(train, d), split_of(valid, d), split_of(test, d)
        s["known_indication"] = s["drug_idx"].isin(k_all)
        s["disease_idx"] = d
        s["drug_id"] = s["drug_idx"].map(ids["idx2id_drug"])
        s["drug_name"] = s["drug_idx"].map(drug_name)
        s.to_parquet(os.path.join(args.out, f"txgnn_{row.mondo_id.replace(':', '_')}.parquet"))
        known_ranks = s.loc[s["known_indication"], "rank"]
        top = ", ".join(s.sort_values("rank").head(8)["drug_name"])
        rows.append(
            {
                "disease": row.name,
                "n_known": len(k_all),
                "in_train": len(k_train),
                "in_valid": len(k_valid),
                "in_test": len(k_test),
                "median_rank_known": float(known_ranks.median()) if len(known_ranks) else np.nan,
                "best_rank_known": int(known_ranks.min()) if len(known_ranks) else -1,
                "top8": top,
            }
        )
    report = pd.DataFrame(rows)
    pd.set_option("display.width", 250, "display.max_colwidth", 90)
    print(report.to_string(index=False), flush=True)
    report.to_csv(os.path.join(args.out, "txgnn_smoke_report.csv"), index=False)
    stamp("done", t0)


if __name__ == "__main__":
    main()
