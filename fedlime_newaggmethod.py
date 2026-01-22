# fedlime_newaggmethod.py
# FedLIME-only (NO fairness, NO DP) using your SAME data pipeline (load_dataset/get_data).
#
# Training:
#   - FedAvg with logits model + BCEWithLogitsLoss
#   - mini-batch DataLoader
#   - client-size-weighted averaging (FedAvg)
#
# LIME:
#   - Runs ONLY at the END of training (after all FedAvg rounds)
#   - Runs locally on each client
#   - Server aggregates ONLY summary stats (mean absolute LIME weights) using FedAvg weights (n_client)
#
# LIME actions (choose ONE):
#   --lime_action none      -> no LIME
#   --lime_action aggregate -> FedAvg aggregate + similarity + prints + saves .npy
#   --lime_action noise     -> same as aggregate, then add Gaussian noise to GLOBAL weights
#   --lime_action rank      -> aggregate (optionally with noise) + print/save top-k ranking
#   --lime_action bins      -> aggregate (optionally with noise) + print/save High/Medium/Low bins
#   --lime_action all       -> aggregate + (optional noise) + ranking + bins

import os
import argparse
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
from lime.lime_tabular import LimeTabularExplainer

from utilities_trustfed import all_metrics
from load_data_trustfed import get_data, load_dataset


def parse_args():
    p = argparse.ArgumentParser(
        description="FedLIME-only: FedAvg training + local LIME per client + FedAvg aggregation of explanation summaries."
    )

    p.add_argument("--num_clients", type=int, default=3, choices=[3, 5, 10, 15])

    p.add_argument(
        "--dataset_name",
        type=str,
        default="bank",
        choices=[
            "adult", "adult-age",
            "bank", "bank-age", "bank-age-5",
            "law", "law-income",
            "default", "default-age",
            "acs",
        ],
    )

    # FedAvg training
    p.add_argument("--epochs", type=int, default=2, help="Local epochs per client per communication round")
    p.add_argument("--communication_rounds", type=int, default=5, help="Number of FedAvg rounds")
    p.add_argument("--batch_size", type=int, default=128, help="Mini-batch size for local training")
    p.add_argument("--lr", type=float, default=1e-3, help="Client learning rate")
    p.add_argument("--seed", type=int, default=0, help="Random seed")

    # imbalance option
    p.add_argument(
        "--use_pos_weight",
        type=str,
        default="yes",
        choices=["yes", "no"],
        help="Use BCEWithLogitsLoss(pos_weight=neg/pos) per client. Default yes.",
    )

    # LIME action
    p.add_argument(
        "--lime_action",
        type=str,
        default="all",
        choices=["none", "aggregate", "noise", "rank", "bins", "all"],
        help="What to do with LIME at the end.",
    )

    p.add_argument("--lime_instances", type=int, default=100, help="How many instances to explain per client")
    p.add_argument("--lime_num_features", type=int, default=10, help="How many features LIME returns per instance")
    p.add_argument("--lime_num_samples", type=int, default=2000, help="LIME perturbation samples per instance")

    # noise on aggregated global weights
    p.add_argument("--lime_noise_std", type=float, default=0.0, help="Std-dev of Gaussian noise for global weights")
    p.add_argument(
        "--lime_noise_clip",
        type=str,
        default="yes",
        choices=["yes", "no"],
        help="After noise, clip global mean-abs weights to be non-negative. Default yes.",
    )

    # ranking
    p.add_argument("--top_k", type=int, default=10, help="Top-K features for ranking/printing")

    # bins
    p.add_argument(
        "--bin_method",
        type=str,
        default="quantile",
        choices=["quantile", "fixed"],
        help="How to create High/Medium/Low bins from global mean-abs weights.",
    )
    p.add_argument("--bin_high", type=float, default=0.66, help="Quantile for High (if bin_method=quantile)")
    p.add_argument("--bin_med", type=float, default=0.33, help="Quantile for Medium (if bin_method=quantile)")
    p.add_argument("--bin_fixed_high", type=float, default=0.01, help="Fixed threshold for High (if bin_method=fixed)")
    p.add_argument("--bin_fixed_med", type=float, default=0.005, help="Fixed threshold for Medium (if bin_method=fixed)")

    # output folder
    p.add_argument("--out_dir", type=str, default="fedlime_newaggmethod", help="Subfolder name for outputs")

    return p.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_model(input_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),  # logits
    )


def make_predict_proba_from_logits(model: nn.Module, device):
    model.eval()

    def predict_proba(x_np: np.ndarray) -> np.ndarray:
        x = torch.tensor(x_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = model(x).squeeze(-1)
            p1 = torch.sigmoid(logits).detach().cpu().numpy()
        p1 = np.clip(p1, 1e-7, 1.0 - 1e-7)
        p0 = 1.0 - p1
        return np.vstack([p0, p1]).T

    return predict_proba


def client_lime_summary(
    client_name: str,
    model: nn.Module,
    clients_data,
    column_names_list,
    num_instances: int,
    num_features_in_exp: int,
    num_samples: int,
    device,
    seed: int,
):
    X1, y1, s1, y1_potential = get_data(client_name, clients_data)
    X_np = X1.detach().cpu().numpy()
    n_client = int(X_np.shape[0])

    rng = np.random.default_rng(seed)
    n_explain = min(num_instances, n_client)
    if n_explain <= 0:
        return {
            "client": client_name,
            "n_client": n_client,
            "n_explained": 0,
            "mean_abs": np.zeros(X_np.shape[1], dtype=np.float64),
            "mean_signed": np.zeros(X_np.shape[1], dtype=np.float64),
        }

    idx = rng.choice(n_client, size=n_explain, replace=False)
    predict_proba = make_predict_proba_from_logits(model, device)

    explainer = LimeTabularExplainer(
        training_data=X_np,
        feature_names=column_names_list,
        class_names=["0", "1"],
        mode="classification",
        discretize_continuous=True,
    )

    n_features = X_np.shape[1]
    W = []
    for i in idx:
        exp = explainer.explain_instance(
            data_row=X_np[i],
            predict_fn=predict_proba,
            num_features=num_features_in_exp,
            num_samples=num_samples,
        )
        weights_for_class1 = dict(exp.as_map()[1])
        w = np.zeros(n_features, dtype=np.float64)
        for feat_idx, weight in weights_for_class1.items():
            w[int(feat_idx)] = float(weight)
        W.append(w)

    W = np.vstack(W)
    return {
        "client": client_name,
        "n_client": n_client,
        "n_explained": int(W.shape[0]),
        "mean_abs": np.mean(np.abs(W), axis=0),
        "mean_signed": np.mean(W, axis=0),
    }


def fedavg_aggregate_mean_abs(client_mean_abs: np.ndarray, client_sizes: np.ndarray) -> np.ndarray:
    sizes = client_sizes.astype(np.float64)
    denom = sizes.sum()
    if denom <= 0:
        w = np.ones_like(sizes) / len(sizes)
    else:
        w = sizes / denom
    return (client_mean_abs * w[:, None]).sum(axis=0)


def make_feature_ranking(column_names_list, global_mean_abs, top_k: int):
    idx = np.argsort(-global_mean_abs)[:top_k]
    return [(column_names_list[i], float(global_mean_abs[i])) for i in idx]


def make_bins(column_names_list, global_mean_abs, method: str, q_high: float, q_med: float, fixed_high: float, fixed_med: float):
    vals = np.asarray(global_mean_abs, dtype=np.float64)

    if method == "quantile":
        nonzero = vals[vals > 0]
        if nonzero.size == 0:
            th_high, th_med = 0.0, 0.0
        else:
            th_high = float(np.quantile(nonzero, q_high))
            th_med = float(np.quantile(nonzero, q_med))
    else:
        th_high = float(fixed_high)
        th_med = float(fixed_med)

    rows = []
    for name, v in zip(column_names_list, vals):
        if v >= th_high:
            b = "High"
        elif v >= th_med:
            b = "Medium"
        else:
            b = "Low"
        rows.append((name, float(v), b))

    return rows, th_high, th_med


def print_lime_summary(column_names, client_names, client_sizes, client_mean_abs, global_mean_abs, sim, top_k=10):
    sizes = client_sizes.astype(np.float64)
    w = sizes / (sizes.sum() + 1e-12)

    print("\n[FedLIME] Client weights (FedAvg):")
    for name, n, wi in zip(client_names, client_sizes, w):
        print(f"  {name:>10s}  n={int(n):>6d}  weight={wi:.4f}")

    idx = np.argsort(-global_mean_abs)[:top_k]
    print(f"\n[FedLIME] Top {top_k} GLOBAL features (mean|weight|):")
    for i in idx:
        print(f"  {column_names[i]:<25s} {global_mean_abs[i]:.6f}")

    print(f"\n[FedLIME] Top {top_k} per CLIENT (mean|weight|):")
    for c, cname in enumerate(client_names):
        c_idx = np.argsort(-client_mean_abs[c])[:top_k]
        print(f"  -- {cname} --")
        for i in c_idx:
            print(f"     {column_names[i]:<25s} {client_mean_abs[c, i]:.6f}")

    print("\n[FedLIME] Cosine similarity (client mean_abs):")
    sim_round = np.round(sim, 3)
    header = " " * 12 + " ".join([f"{c:>8s}" for c in client_names])
    print(header)
    for i, cname in enumerate(client_names):
        row = " ".join([f"{sim_round[i, j]:8.3f}" for j in range(len(client_names))])
        print(f"{cname:>10s}  {row}")


def print_bins(rows):
    from collections import Counter

    counts = Counter([b for _, _, b in rows])
    print("\n[FedLIME] Bin counts:", dict(counts))

    for bname in ["High", "Medium", "Low"]:
        feats = [(f, v) for f, v, b in rows if b == bname]
        feats = sorted(feats, key=lambda x: -x[1])[:10]
        print(f"\nTop in {bname}:")
        for f, v in feats:
            print(f"  {f:<25s} {v:.6f}")


def federated_lime_report(
    model: nn.Module,
    clients_data,
    column_names_list,
    num_instances: int,
    num_features_in_exp: int,
    num_samples: int,
    device,
    seed: int,
    noise_std: float,
    noise_clip_nonneg: bool,
):
    summaries = []
    for client_name in clients_data.keys():
        summaries.append(
            client_lime_summary(
                client_name=client_name,
                model=model,
                clients_data=clients_data,
                column_names_list=column_names_list,
                num_instances=num_instances,
                num_features_in_exp=num_features_in_exp,
                num_samples=num_samples,
                device=device,
                seed=seed,
            )
        )

    client_names = [s["client"] for s in summaries]
    client_mean_abs = np.vstack([s["mean_abs"] for s in summaries])
    client_sizes = np.array([s["n_client"] for s in summaries], dtype=np.float64)

    global_mean_abs = fedavg_aggregate_mean_abs(client_mean_abs, client_sizes)

    if noise_std and float(noise_std) > 0.0:
        global_mean_abs = global_mean_abs + np.random.normal(0.0, float(noise_std), size=global_mean_abs.shape)
        if noise_clip_nonneg:
            global_mean_abs = np.clip(global_mean_abs, 0.0, None)

    sim = cosine_similarity(client_mean_abs) if client_mean_abs.shape[0] >= 2 else np.ones((client_mean_abs.shape[0], client_mean_abs.shape[0]))

    return {
        "client_names": client_names,
        "client_sizes": client_sizes,
        "client_mean_abs": client_mean_abs,
        "global_mean_abs": global_mean_abs,
        "similarity_matrix": sim,
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cpu")

    dataset_name = args.dataset_name

    if dataset_name in ["adult", "adult-age"]:
        url = "./datasets/adult.csv"
        sensitive_feature = "sex"
    elif dataset_name in ["bank", "bank-age", "bank-age-5"]:
        url = "./datasets/bank-full.csv"
        sensitive_feature = "marital"
    elif dataset_name in ["law", "law-income"]:
        url = "./datasets/law.csv"
        sensitive_feature = "sex"
    elif dataset_name in ["default", "default-age"]:
        url = "./datasets/default.csv"
        sensitive_feature = "SEX"
    elif dataset_name == "acs":
        url = "./datasets/acs/"
        sensitive_feature = "sex"
    else:
        raise ValueError("dataset not supported, please update file load_data_trustfed.py")

    clients_data, X_test, y_test, sex_list, column_names_list, ytest_potential = load_dataset(
        url, dataset_name, args.num_clients, sensitive_feature
    )

    actual_clients = len(clients_data)
    if actual_clients != args.num_clients:
        print(
            f"\n[NOTE] You requested num_clients={args.num_clients}, "
            f"but load_dataset returned {actual_clients} clients for dataset_name='{dataset_name}'."
        )
        print("This is normal for some non-IID loaders that hardcode groups.\n")

    X_test = X_test.to(device).float()
    y_test = y_test.to(device).float()

    global_model = create_model(X_test.shape[1]).to(device)

    acc_list, bal_acc_list, auprc_list = [], [], []

    def train_one_round(lr: float):
        total_samples = 0
        params_sum = [torch.zeros_like(p.data) for p in global_model.parameters()]

        for client_name in clients_data.keys():
            X1, y1, s1, y1_potential = get_data(client_name, clients_data)
            X1 = X1.to(device).float()
            y1 = y1.to(device).float().view(-1)

            ds = TensorDataset(X1, y1)
            dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

            model1 = create_model(X1.shape[1]).to(device)
            model1.load_state_dict(global_model.state_dict())
            optimizer1 = optim.Adam(model1.parameters(), lr=float(lr))

            if args.use_pos_weight == "yes":
                pos = float((y1 == 1).sum().item())
                neg = float((y1 == 0).sum().item())
                if pos > 0:
                    pos_weight = torch.tensor([neg / (pos + 1e-12)], device=device)
                else:
                    pos_weight = torch.tensor([1.0], device=device)
                loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                loss_fn = nn.BCEWithLogitsLoss()

            model1.train()
            for _ in range(args.epochs):
                for xb, yb in dl:
                    optimizer1.zero_grad()
                    logits = model1(xb).view(-1)
                    loss = loss_fn(logits, yb)
                    loss.backward()
                    optimizer1.step()

            n = int(X1.shape[0])
            total_samples += n
            for p, p_sum in zip(model1.parameters(), params_sum):
                p_sum.add_(p.data * n)

        with torch.no_grad():
            for p_global, p_sum in zip(global_model.parameters(), params_sum):
                p_global.copy_(p_sum / float(total_samples))

    def evaluate_global():
        global_model.eval()
        with torch.no_grad():
            logits = global_model(X_test).squeeze()
            y_prob = torch.sigmoid(logits)

            _, _, bal_acc, _, _, _, _, _, acc, auc = all_metrics(y_test.cpu(), y_prob.cpu())
            auprc = average_precision_score(y_test.cpu(), y_prob.cpu())

        return float(acc), float(bal_acc), float(auprc)

    for r in range(args.communication_rounds):
        print(f"\nFedAvg round {r+1}/{args.communication_rounds}")
        train_one_round(lr=args.lr)

        acc, bal_acc, auprc = evaluate_global()
        acc_list.append(acc)
        bal_acc_list.append(bal_acc)
        auprc_list.append(auprc)
        print(f"  acc={acc:.4f}  bal_acc={bal_acc:.4f}  auprc={auprc:.4f}")

    destination = f"./results/{dataset_name}/{args.out_dir}/"
    os.makedirs(destination, exist_ok=True)

    np.save(os.path.join(destination, "acc_curve.npy"), np.array(acc_list, dtype=np.float64))
    np.save(os.path.join(destination, "bal_acc_curve.npy"), np.array(bal_acc_list, dtype=np.float64))
    np.save(os.path.join(destination, "auprc_curve.npy"), np.array(auprc_list, dtype=np.float64))

    if args.lime_action == "none":
        print(f"\nDone. Saved results to: {destination}")
        return

    want_noise = (args.lime_action in ["noise", "all"]) or (args.lime_noise_std > 0 and args.lime_action in ["rank", "bins"])
    noise_std = float(args.lime_noise_std) if want_noise else 0.0
    noise_clip_nonneg = (args.lime_noise_clip == "yes")

    print("\nRunning FedLIME (LIME local per client; FedAvg aggregation of summary stats)...")
    lime_results = federated_lime_report(
        model=global_model,
        clients_data=clients_data,
        column_names_list=column_names_list,
        num_instances=args.lime_instances,
        num_features_in_exp=args.lime_num_features,
        num_samples=args.lime_num_samples,
        device=device,
        seed=args.seed,
        noise_std=noise_std,
        noise_clip_nonneg=noise_clip_nonneg,
    )

    client_names = lime_results["client_names"]
    client_sizes = lime_results["client_sizes"]
    client_mean_abs = lime_results["client_mean_abs"]
    global_mean_abs = lime_results["global_mean_abs"]
    sim = lime_results["similarity_matrix"]

    if noise_std > 0:
        print(f"\n[FedLIME] Gaussian noise enabled on GLOBAL weights: std={noise_std} (clip_nonneg={noise_clip_nonneg})")

    # Always print (so you don't have to read .npy)
    print_lime_summary(
        column_names=column_names_list,
        client_names=client_names,
        client_sizes=client_sizes,
        client_mean_abs=client_mean_abs,
        global_mean_abs=global_mean_abs,
        sim=sim,
        top_k=args.top_k,
    )

    # Always save arrays too
    np.save(os.path.join(destination, "lime_global_mean_abs.npy"), global_mean_abs)
    np.save(os.path.join(destination, "lime_client_mean_abs.npy"), client_mean_abs)
    np.save(os.path.join(destination, "lime_similarity.npy"), sim)
    np.save(os.path.join(destination, "lime_client_names.npy"), np.array(client_names, dtype=object))
    np.save(os.path.join(destination, "lime_client_sizes.npy"), np.array(client_sizes, dtype=np.float64))

    # Ranking CSV
    if args.lime_action in ["rank", "all", "noise"]:
        ranking = make_feature_ranking(column_names_list, global_mean_abs, top_k=args.top_k)
        print(f"\n[FedLIME] Top {args.top_k} GLOBAL features (saved to CSV too):")
        for fname, val in ranking:
            print(f"  {fname}: {val:.6f}")

        rank_path = os.path.join(destination, "lime_feature_ranking.csv")
        with open(rank_path, "w", encoding="utf-8") as f:
            f.write("feature,mean_abs_weight\n")
            for fname, val in ranking:
                f.write(f"{fname},{val}\n")

    # Bins CSV
    if args.lime_action in ["bins", "all"]:
        rows, th_high, th_med = make_bins(
            column_names_list,
            global_mean_abs,
            method=args.bin_method,
            q_high=args.bin_high,
            q_med=args.bin_med,
            fixed_high=args.bin_fixed_high,
            fixed_med=args.bin_fixed_med,
        )
        print(f"\n[FedLIME] Binning thresholds -> High >= {th_high:.6f}, Medium >= {th_med:.6f} (method={args.bin_method})")
        print_bins(rows)

        bins_path = os.path.join(destination, "lime_feature_bins.csv")
        with open(bins_path, "w", encoding="utf-8") as f:
            f.write("feature,mean_abs_weight,bin\n")
            for feat, val, b in rows:
                f.write(f"{feat},{val},{b}\n")

    print(f"\nDone. Saved results to: {destination}")


if __name__ == "__main__":
    main()
