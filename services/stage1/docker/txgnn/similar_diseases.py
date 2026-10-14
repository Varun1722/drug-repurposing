"""List the diseases TxGNN's prototype step borrows from, for each test disease.

TxGNN (proto=True, proto_num=3, sim_measure='all_nodes_profile') mixes a disease's own embedding
with the similarity-weighted average of its 3 most similar diseases that have known indications.
The mix weight comes from `rarity`: 0.7 * exp(-0.7 * n) + 0.2, where n is the number of known
drugs, so a disease with none borrows 0.9 and one with ten or more borrows about 0.2.
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch


def txgnn_key(node_id):
    if "_" in node_id:
        return node_id
    try:
        return str(float(node_id))
    except ValueError:
        return node_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/primekg")
    ap.add_argument("--weights", default="/weights/model_ckpt")
    ap.add_argument("--diseases", default="/config/test_diseases.csv")
    ap.add_argument("--out", default="/out")
    args = ap.parse_args()
    t0 = time.time()
    from txgnn import TxData, TxGNN

    data = TxData(data_folder_path=args.data)
    data.prepare_split(split="full_graph", seed=1)
    model = TxGNN(data=data, weight_bias_track=False, proj_name="x", exp_name="x", device="cpu")
    model.load_pretrained(args.weights)
    print(f"[{time.time() - t0:.0f}s] model loaded", flush=True)

    from txgnn.utils import obtain_disease_profile, sim_matrix

    ids = data.retrieve_id_mapping()
    pred = model.model.pred
    etype = ("drug", "indication", "disease")  # the relation used when we score drug -> disease
    G = model.G
    print("similarity matrices:", {k[1]: tuple(v.shape) for k, v in pred.sim_all_etypes.items()}, flush=True)
    sim_all = pred.sim_all_etypes[etype]
    d2id = pred.diseaseid2id_etypes[etype]  # disease idx -> row in sim_all
    keys = torch.where(G.in_degrees(etype=etype) != 0)[0]  # diseases with known indications
    in_deg = G.in_degrees(etype=etype)
    profiles = pred.diseases_profile_etypes[etype]
    print(f"key diseases (with known indications): {len(keys)}; profiles cached: {len(profiles)}", flush=True)
    idx2id, id2name = ids["idx2id_disease"], ids["id2name_disease"]

    def name(i):
        return id2name.get(idx2id.get(i), str(i))

    by_key = {v: k for k, v in idx2id.items()}
    table = pd.read_csv(args.diseases, dtype={"txgnn_node_id": str}).dropna(subset=["txgnn_node_id"])
    disease_etypes, disease_nodes = ["disease_disease", "rev_disease_protein"], ["disease", "gene/protein"]
    rows = []
    for r in table.itertuples():
        d = by_key.get(txgnn_key(r.txgnn_node_id))
        if d is None:
            continue
        d = int(d)
        n_known = int(in_deg[d])
        if d in d2id:
            sim = sim_all[d2id[d]].clone().numpy()
            how = "stored similarity row"
        else:
            q = obtain_disease_profile(G, torch.tensor(d), disease_etypes, disease_nodes)
            kp = torch.cat([profiles[int(k)] for k in keys]).view(len(keys), -1)
            sim = sim_matrix(q.view(1, -1), kp)[0].numpy()
            how = "profile computed on the fly (disease has no known indications)"
        top = np.argsort(-sim)[:3]  # the model takes the top 3 as they are, including itself
        picks = [(name(int(keys[j])), float(sim[j])) for j in top]
        rows.append(
            {
                "disease": r.name,
                "known_indications": n_known,
                "borrow_weight": round(0.7 * float(np.exp(-0.7 * n_known)) + 0.2, 3),
                "similar_diseases_used": "; ".join(f"{n} ({s:.2f})" for n, s in picks),
                "how": how,
            }
        )
    report = pd.DataFrame(rows)
    pd.set_option("display.width", 250, "display.max_colwidth", 140)
    print(report.to_string(index=False), flush=True)
    report.to_csv(f"{args.out}/txgnn_similar_diseases.csv", index=False)


if __name__ == "__main__":
    main()
