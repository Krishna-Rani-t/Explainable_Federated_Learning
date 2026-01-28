# fedlime.py
# FedLIME-only (NO fairness, NO DP) using your SAME data pipeline (load_dataset/get_data).
# Training: FedAvg with logits model + BCEWithLogitsLoss + mini-batch DataLoader + client-size-weighted averaging.
#
# IID datasets:
#   adult, bank, law  -> load_dataset() does random client splits using num_clients
#
# non-IID datasets (attribute-based, handled INSIDE load_dataset):
#   adult-age, bank-age, bank-age-5, law-income, default-age
#   NOTE: these are often hard-coded to a fixed number of clients (e.g., 3 or 5).
#
# Example runs:
#   IID:
#     python fedlime.py --dataset_name bank --num_clients 10 --communication_rounds 20 --epochs 2 --lr 0.001
#   non-IID:
#     python fedlime.py --dataset_name adult-age --num_clients 3 --communication_rounds 20 --epochs 2 --lr 0.001

import os
import argparse
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
from lime.lime_tabular import LimeTabularExplainer

from utilities_fedlime import all_metrics
from load_data_trustfed import get_data, load_dataset


def parse_args():
    p = argparse.ArgumentParser(
        description="FedLIME-only (no fairness, no DP): FedAvg training + local LIME per client + aggregated stats."
    )

    p.add_argument("--num_clients", type=int, default=3, choices=[3, 5, 10, 15])

    # ✅ UPDATED: include non-IID dataset variants that your load_dataset already supports
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

    p.add_argument("--epochs", type=int, default=2, help="Local epochs per client per communication round")
    p.add_argument("--communication_rounds", type=int, default=5, help="Number of FedAvg rounds")
    p.add_argument("--batch_size", type=int, default=128, help="Mini-batch size for local training")
    p.add_argument("--lr", type=float, default=1e-3, help="Client learning rate")

    # Optional: helps with imbalanced datasets (like bank)
    p.add_argument(
        "--use_pos_weight",
        type=str,
        default="yes",
        choices=["yes", "no"],
        help="Use BCEWithLogitsLoss(pos_weight=neg/pos) per client. Default yes.",
    )

    # FedLIME options
    p.add_argument("--run_lime", type=str, default="yes", choices=["yes", "no"])
    p.add_argument("--lime_instances", type=int, default=100)
    p.add_argument("--lime_num_features", type=int, default=10)
    p.add_argument("--lime_num_samples", type=int, default=2000)

    return p.parse_args()


# -----------------------------
# Model (LOGITS output, no Sigmoid)
# -----------------------------
def create_model(input_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),  # logits
    )


# -----------------------------
# LIME helpers
# -----------------------------
def make_predict_proba_from_logits(model: nn.Module, device):
    model.eval()

    def predict_proba(x_np: np.ndarray) -> np.ndarray:
        x = torch.tensor(x_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = model(x).squeeze(-1)
            p1 = torch.sigmoid(logits).detach().cpu().numpy()
        p1 = np.clip(p1, 1e-7, 1 - 1e-7)
        p0 = 1.0 - p1
        return np.vstack([p0, p1]).T  # (n,2)

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
    seed: int = 0,
):
    # local-only client data
    X1, y1, s1, y1_potential = get_data(client_name, clients_data)
    X_np = X1.detach().cpu().numpy()

    rng = np.random.default_rng(seed)
    n = min(num_instances, len(X_np))
    if n <= 0:
        return {
            "client": client_name,
            "n_explained": 0,
            "mean_abs": np.zeros(X_np.shape[1], dtype=np.float64),
            "mean_signed": np.zeros(X_np.shape[1], dtype=np.float64),
        }

    idx = rng.choice(len(X_np), size=n, replace=False)
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
        weights_for_class1 = dict(exp.as_map()[1])  # {feature_index: weight}
        w = np.zeros(n_features, dtype=np.float64)
        for feat_idx, weight in weights_for_class1.items():
            w[int(feat_idx)] = float(weight)
        W.append(w)

    W = np.vstack(W)
    return {
        "client": client_name,
        "n_explained": int(W.shape[0]),
        "mean_abs": np.mean(np.abs(W), axis=0),
        "mean_signed": np.mean(W, axis=0),
    }


def federated_lime_report(
    model: nn.Module,
    clients_data,
    column_names_list,
    num_instances: int,
    num_features_in_exp: int,
    num_samples: int,
    device,
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
                seed=0,
            )
        )

    client_names = [s["client"] for s in summaries]
    M = np.vstack([s["mean_abs"] for s in summaries])  # (C, F)

    client_n_explained = [s["n_explained"] for s in summaries]
    weights = np.array([s["n_explained"] for s in summaries], dtype=np.float64)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    weights = weights / weights.sum()

    global_mean_abs = (M * weights[:, None]).sum(axis=0)
    sim = cosine_similarity(M) if M.shape[0] >= 2 else np.ones((M.shape[0], M.shape[0]))

    top_idx = np.argsort(-global_mean_abs)[:10]
    top_features = [(column_names_list[i], float(global_mean_abs[i])) for i in top_idx]

    return {
        "client_names": client_names,
        "client_mean_abs": M,
        "client_n_explained": client_n_explained,        
        "global_mean_abs": global_mean_abs,
        "similarity_matrix": sim,
        "top_features": top_features,
    }

def client_age_split_str(client_name, clients_data):
    meta = clients_data.get(client_name, {}).get("meta", {})
    if "age_min" in meta and "age_max" in meta:
        return f"{meta['age_min']}-{meta['age_max']}"
    return "N/A"

def main():
    args = parse_args()
    device = torch.device("cpu")

    # -----------------------------
    # Dataset routing (UPDATED to support -age / -income variants)
    # -----------------------------
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

    # -----------------------------
    # Load data (UNCHANGED)
    # -----------------------------
    clients_data, X_test, y_test, sex_list, column_names_list, ytest_potential = load_dataset(
        url, dataset_name, args.num_clients, sensitive_feature
    )

    # ✅ helpful warning (training only; not changing preprocessing)
    actual_clients = len(clients_data)
    if actual_clients != args.num_clients:
        print(
            f"\n[NOTE] You requested num_clients={args.num_clients}, "
            f"but load_dataset returned {actual_clients} clients for dataset_name='{dataset_name}'."
        )
        print("This is normal for some non-IID loaders (e.g., adult-age often hardcodes 3 age groups).\n")

    X_test = X_test.to(device).float()
    y_test = y_test.to(device).float()

    global_model = create_model(X_test.shape[1]).to(device)

    acc_list, bal_acc_list, auprc_list = [], [], []

    # -----------------------------
    # Training: FedAvg (client-size-weighted)
    # -----------------------------
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

            # Optional pos_weight for imbalance
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

            # weighted FedAvg accumulate
            n = X1.shape[0]
            total_samples += n
            for p, p_sum in zip(model1.parameters(), params_sum):
                p_sum.add_(p.data * n)

        with torch.no_grad():
            for p_global, p_sum in zip(global_model.parameters(), params_sum):
                p_global.copy_(p_sum / total_samples)

    def evaluate_global():
        global_model.eval()
        with torch.no_grad():
            logits = global_model(X_test).squeeze()
            y_prob = torch.sigmoid(logits)

            # your existing metric helper
            _, _, bal_acc, _, _, _, _, _, acc, auc = all_metrics(y_test.cpu(), y_prob.cpu())
            auprc = average_precision_score(y_test.cpu(), y_prob.cpu())

        return float(acc), float(bal_acc), float(auprc)

    for r in range(args.communication_rounds):
        print(f"\nCommunication round {r+1}/{args.communication_rounds}")
        train_one_round(lr=args.lr)

        acc, bal_acc, auprc = evaluate_global()
        acc_list.append(acc)
        bal_acc_list.append(bal_acc)
        auprc_list.append(auprc)

        print(f"  acc={acc:.4f}  bal_acc={bal_acc:.4f}  auprc={auprc:.4f}")

    # -----------------------------
    # Save curves + FedLIME
    # -----------------------------
    destination = f"./results/{dataset_name}/fedlime_only_clean/"
    os.makedirs(destination, exist_ok=True)

    np.save(os.path.join(destination, "acc_curve.npy"), np.array(acc_list, dtype=np.float64))
    np.save(os.path.join(destination, "bal_acc_curve.npy"), np.array(bal_acc_list, dtype=np.float64))
    np.save(os.path.join(destination, "auprc_curve.npy"), np.array(auprc_list, dtype=np.float64))

    if args.run_lime == "yes":
        print("\nRunning FedLIME (local per client, aggregated stats only)...")
        lime_results = federated_lime_report(
            model=global_model,
            clients_data=clients_data,
            column_names_list=column_names_list,
            num_instances=args.lime_instances,
            num_features_in_exp=args.lime_num_features,
            num_samples=args.lime_num_samples,
            device=device,
        )

        print("\nTop 10 global LIME features (mean abs weight):")
        for fname, val in lime_results["top_features"]:
            print(f"{fname}: {val:.6f}")

        print("\nPer-client LIME (Top features by mean abs weight):")
        top_k = args.lime_num_features  # or set e.g. 10

        for c_idx, cname in enumerate(lime_results["client_names"]):
            mean_abs = lime_results["client_mean_abs"][c_idx]
            n_exp = lime_results["client_n_explained"][c_idx]

            top_idx = np.argsort(-mean_abs)[:top_k]

            # optional: show client size + pos rate
            Xc, yc, _, _ = get_data(cname, clients_data)
            pos_rate = float((yc == 1).float().mean().item())
            age_split = client_age_split_str(cname, clients_data)
            print(f"\n[{cname}] age_split={age_split} | n_explained={n_exp} | n_samples={Xc.shape[0]} | pos_rate={pos_rate:.3f}")

            print(f"\n[{cname}] n_explained={n_exp} | n_samples={Xc.shape[0]} | pos_rate={pos_rate:.3f}")
            for rank, j in enumerate(top_idx, start=1):
                print(f"  {rank:>2}. {column_names_list[j]}: {mean_abs[j]:.6f}")

        np.save(os.path.join(destination, "lime_global_mean_abs.npy"), lime_results["global_mean_abs"])
        np.save(os.path.join(destination, "lime_client_mean_abs.npy"), lime_results["client_mean_abs"])
        np.save(os.path.join(destination, "lime_similarity.npy"), lime_results["similarity_matrix"])
        np.save(os.path.join(destination, "lime_client_names.npy"), np.array(lime_results["client_names"], dtype=object))

    print(f"\nDone. Saved results to: {destination}")


if __name__ == "__main__":
    main()
