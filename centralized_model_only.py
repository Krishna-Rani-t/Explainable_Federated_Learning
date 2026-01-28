# global_model_only.py
# Train a SINGLE global (centralized) model using your SAME data pipeline (load_dataset/get_data).
# No federated learning / no client averaging.
#
# - Uses logits model + BCEWithLogitsLoss (stable)
# - Optional pos_weight for imbalanced data (default: yes)
# - Optional LIME on the global model (explains global instances)
#
# Run example:
#   python global_model_only.py --dataset_name bank --num_clients 3 --epochs 5 --batch_size 128 --lr 0.001 --run_lime yes --lime_instances 50

import os
import argparse
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import average_precision_score, r2_score
from lime.lime_tabular import LimeTabularExplainer

from utilities_fedlime import all_metrics
from load_data_trustfed import get_data, load_dataset


def parse_args():
    p = argparse.ArgumentParser(
        description="Global model only (no federated): train centrally using load_dataset/get_data, optionally run LIME."
    )
    p.add_argument("--dataset_name", type=str, default="bank", choices=["adult", "default", "acs", "bank", "law"])
    # Still required because your load_dataset(...) expects it and builds the same train/test split logic
    p.add_argument("--num_clients", type=int, default=3, choices=[3, 5, 10, 15])

    p.add_argument("--epochs", type=int, default=5, help="Training epochs for the global model")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)

    p.add_argument("--use_pos_weight", type=str, default="yes", choices=["yes", "no"],
                   help="Use BCEWithLogitsLoss(pos_weight=neg/pos) to handle imbalance. Default yes.")

    # LIME options
    p.add_argument("--run_lime", type=str, default="yes", choices=["yes", "no"])
    p.add_argument("--lime_instances", type=int, default=100, help="How many training instances to explain with LIME")
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


def make_predict_proba_from_logits(model: nn.Module, device):
    model.eval()

    def predict_proba(x_np: np.ndarray) -> np.ndarray:
        x = torch.tensor(x_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = model(x).squeeze(-1)
            p1 = torch.sigmoid(logits).detach().cpu().numpy()
        p1 = np.clip(p1, 1e-7, 1 - 1e-7)
        p0 = 1.0 - p1
        return np.vstack([p0, p1]).T  # (n, 2)

    return predict_proba


def compute_global_fidelity(
    model: nn.Module,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    global_mean_abs: np.ndarray,
    device,
):
    model.eval()
    X_np = X_test.detach().cpu().numpy()
    
    with torch.no_grad():
        logits = model(X_test).squeeze()
        y_prob_actual = torch.sigmoid(logits).cpu().numpy()
    
    X_normalized = (X_np - X_np.mean(axis=0)) / (X_np.std(axis=0) + 1e-10)
    lime_predictions = np.dot(X_normalized, global_mean_abs)
    
    lime_predictions = (lime_predictions - lime_predictions.min()) / (lime_predictions.max() - lime_predictions.min() + 1e-10)
    
    r2 = r2_score(y_prob_actual, lime_predictions)
    
    return float(r2)


def main():
    args = parse_args()
    device = torch.device("cpu")

    # -----------------------------
    # Dataset routing (UNCHANGED)
    # -----------------------------
    dataset_name = args.dataset_name
    if dataset_name == "adult":
        url = "./datasets/adult.csv"
        sensitive_feature = "sex"
    elif dataset_name == "bank":
        url = "./datasets/bank-full.csv"
        sensitive_feature = "marital"
    elif dataset_name == "default":
        url = "./datasets/default.csv"
        sensitive_feature = "SEX"
    elif dataset_name == "law":
        url = "./datasets/law.csv"
        sensitive_feature = "sex"
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

    # Build ONE global training set by concatenating all client partitions
    X_train_parts, y_train_parts = [], []
    for client_name in clients_data.keys():
        X1, y1, s1, y1_potential = get_data(client_name, clients_data)
        X_train_parts.append(X1)
        y_train_parts.append(y1.view(-1))

    X_train = torch.cat(X_train_parts, dim=0).to(device).float()
    y_train = torch.cat(y_train_parts, dim=0).to(device).float().view(-1)

    X_test = X_test.to(device).float()
    y_test = y_test.to(device).float().view(-1)

    print(f"Train size: {X_train.shape}, positives={int((y_train==1).sum().item())}")
    print(f"Test size : {X_test.shape}, positives={int((y_test==1).sum().item())}")

    model = create_model(X_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(args.lr))

    # Loss (optional pos_weight)
    if args.use_pos_weight == "yes":
        pos = float((y_train == 1).sum().item())
        neg = float((y_train == 0).sum().item())
        if pos > 0:
            pos_weight = torch.tensor([neg / (pos + 1e-12)], device=device)
        else:
            pos_weight = torch.tensor([1.0], device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Using pos_weight={float(pos_weight.item()):.4f}")
    else:
        loss_fn = nn.BCEWithLogitsLoss()
        print("Using BCEWithLogitsLoss without pos_weight")

    # DataLoader
    ds = TensorDataset(X_train, y_train)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    # -----------------------------
    # Train global model
    # -----------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in dl:
            optimizer.zero_grad()
            logits = model(xb).view(-1)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * xb.size(0)

        avg_loss = running / len(ds)

        # quick eval
        model.eval()
        with torch.no_grad():
            test_logits = model(X_test).view(-1)
            y_prob = torch.sigmoid(test_logits)

            _, _, bal_acc, _, _, _, _, _, acc, auc = all_metrics(y_test.cpu(), y_prob.cpu())
            auprc = average_precision_score(y_test.cpu(), y_prob.cpu())

        print(f"Epoch {epoch}/{args.epochs}  loss={avg_loss:.4f}  acc={float(acc):.4f}  bal_acc={float(bal_acc):.4f}  auprc={float(auprc):.4f}")

    # Final metrics
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test).view(-1)
        y_prob = torch.sigmoid(test_logits)

        _, _, bal_acc, _, _, _, _, _, acc, auc = all_metrics(y_test.cpu(), y_prob.cpu())
        auprc = average_precision_score(y_test.cpu(), y_prob.cpu())

    print("\nFINAL:")
    print(f"acc={float(acc):.6f}  bal_acc={float(bal_acc):.6f}  auprc={float(auprc):.6f}  auc={float(auc):.6f}")

    # -----------------------------
    # Optional LIME on GLOBAL model
    # -----------------------------
    destination = f"./results/{dataset_name}/global_only/"
    os.makedirs(destination, exist_ok=True)

    if args.run_lime == "yes":
        print("\nRunning LIME on the GLOBAL model...")
        X_np = X_train.detach().cpu().numpy()
        n = min(args.lime_instances, X_np.shape[0])
        rng = np.random.default_rng(0)
        idx = rng.choice(X_np.shape[0], size=n, replace=False)

        predict_fn = make_predict_proba_from_logits(model, device)

        explainer = LimeTabularExplainer(
            training_data=X_np,
            feature_names=column_names_list,
            class_names=["0", "1"],
            mode="classification",
            discretize_continuous=True,
        )

        # Aggregate mean abs weights across explained instances (global)
        n_features = X_np.shape[1]
        W = []
        fidelity_scores = [] 
        for i in idx:
            exp = explainer.explain_instance(
                data_row=X_np[i],
                predict_fn=predict_fn,
                num_features=args.lime_num_features,
                num_samples=args.lime_num_samples,
            )

            if len(W) < 5:
                exp.save_to_file(os.path.join(destination, f"lime_instance_{int(i)}.html"))
            
            weights_for_class1 = dict(exp.as_map()[1])
            w = np.zeros(n_features, dtype=np.float64)
            for feat_idx, weight in weights_for_class1.items():
                w[int(feat_idx)] = float(weight)
            W.append(w)
            
            fidelity_scores.append(float(exp.score))

        W = np.vstack(W)
        global_mean_abs = np.mean(np.abs(W), axis=0)
        
        avg_instance_fidelity = np.mean(fidelity_scores)
        
        print("Computing global fidelity on test set...")
        global_fidelity_test = compute_global_fidelity(
            model=model,
            X_test=X_test,
            y_test=y_test,
            global_mean_abs=global_mean_abs,
            device=device,
        )

        top_idx = np.argsort(-global_mean_abs)[:10]
        print("\nTop 10 global LIME features (mean abs weight):")
        for j in top_idx:
            print(f"{column_names_list[j]}: {float(global_mean_abs[j]):.6f}")
        
        print("\n[LIME] Fidelity Scores (R² - how well LIME approximates the model):")
        print(f"  Per-instance average (training): {avg_instance_fidelity:.4f}")
        print(f"  Global (test set): {global_fidelity_test:.4f}")

        # Save results
        np.save(os.path.join(destination, "lime_global_mean_abs.npy"), global_mean_abs)
        np.save(os.path.join(destination, "lime_instance_weights.npy"), W)
        np.save(os.path.join(destination, "lime_global_fidelity.npy"), np.array([global_fidelity_test], dtype=np.float64))

    print(f"\nDone. Saved results to: {destination}")


if __name__ == "__main__":
    main()
