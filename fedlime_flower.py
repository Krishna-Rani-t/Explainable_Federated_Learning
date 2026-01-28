import os
import argparse
import warnings
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict

import flwr as fl
from flwr.common import Metrics

warnings.filterwarnings("ignore", message=".*start_simulation.*deprecated.*")

from sklearn.metrics import average_precision_score, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from lime.lime_tabular import LimeTabularExplainer

from utilities_fedlime import all_metrics
from load_data_trustfed import get_data, load_dataset


def create_model(input_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )


class FedLIMEClient(fl.client.NumPyClient):
    def __init__(self, client_name, clients_data, model, epochs, lr, batch_size, device):
        self.client_name = client_name
        self.clients_data = clients_data
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        
        X, y, s, y_pot = get_data(client_name, clients_data)
        self.X = X.to(device).float()
        self.y = y.to(device).float().view(-1)
        self.n_samples = len(self.X)
    
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        dataset = TensorDataset(self.X, self.y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        pos = float((self.y == 1).sum().item())
        neg = float((self.y == 0).sum().item())
        if pos > 0:
            pos_weight = torch.tensor([neg / (pos + 1e-12)], device=self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()
        
        self.model.train()
        for epoch in range(self.epochs):
            for xb, yb in dataloader:
                optimizer.zero_grad()
                logits = self.model(xb).view(-1)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
        
        return self.get_parameters(config={}), self.n_samples, {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.X).view(-1)
            y_prob = torch.sigmoid(logits)
            
            criterion = nn.BCEWithLogitsLoss()
            loss = float(criterion(logits, self.y).item())
            
            y_pred = (y_prob > 0.5).float()
            accuracy = float((y_pred == self.y).float().mean().item())
        
        return loss, self.n_samples, {"accuracy": accuracy}


def client_fn(cid: str):
    # Flower passes cid as "0", "1", "2", etc.
    # Convert to "client_1", "client_2", "client_3", etc. (1-indexed)
    client_name = f"client_{int(cid) + 1}"
    return FedLIMEClient(
        client_name=client_name,
        clients_data=clients_data_global,
        model=create_model(input_dim_global),
        epochs=args_global.epochs,
        lr=args_global.lr,
        batch_size=args_global.batch_size,
        device=device_global
    )


def evaluate_global_model(server_round, parameters, config):
    params_dict = zip(global_model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    global_model.load_state_dict(state_dict, strict=True)
    
    global_model.eval()
    with torch.no_grad():
        logits = global_model(X_test_global).view(-1)
        y_prob = torch.sigmoid(logits)
        
        criterion = nn.BCEWithLogitsLoss()
        loss = float(criterion(logits, y_test_global).item())
        
        _, _, bal_acc, _, _, _, _, _, acc, auc = all_metrics(
            y_test_global.cpu(), y_prob.cpu()
        )
        auprc = average_precision_score(y_test_global.cpu(), y_prob.cpu())
    
    print(f"Round {server_round}: loss={loss:.4f}, acc={acc:.4f}, bal_acc={bal_acc:.4f}, auprc={auprc:.4f}")
    
    return loss, {"accuracy": float(acc), "balanced_accuracy": float(bal_acc), "auprc": float(auprc)}


def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    return {"accuracy": sum(accuracies) / sum(examples)}


def make_predict_proba_from_logits(model, device):
    model.eval()
    
    def predict_proba(x_np):
        x = torch.tensor(x_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = model(x).squeeze(-1)
            p1 = torch.sigmoid(logits).cpu().numpy()
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
    noise_std: float = 0.0,
    noise_clip_nonneg: bool = True,
):
    X1, y1, s1, y1_potential = get_data(client_name, clients_data)
    X_np = X1.detach().cpu().numpy()
    n_client, n_features = X_np.shape

    n_explain = min(num_instances, n_client)
    if n_explain <= 0:
        return {
            "client": client_name,
            "n_client": n_client,
            "n_explained": 0,
            "mean_abs": np.zeros(n_features, dtype=np.float64),
            "mean_signed": np.zeros(n_features, dtype=np.float64),
            "fidelity_scores": [],
        }

    rng = np.random.default_rng(seed)
    idx = rng.choice(n_client, size=n_explain, replace=False)
    
    predict_proba = make_predict_proba_from_logits(model, device)
    explainer = LimeTabularExplainer(
        training_data=X_np,
        feature_names=column_names_list,
        class_names=["0", "1"],
        mode="classification",
        discretize_continuous=True,
    )

    W = []
    fidelity_scores = []
    for i in idx:
        exp = explainer.explain_instance(
            data_row=X_np[i],
            predict_fn=predict_proba,
            num_features=num_features_in_exp,
            num_samples=num_samples,
        )
        
        w = np.zeros(n_features, dtype=np.float64)
        for feat_idx, weight in exp.as_map()[1]:
            w[feat_idx] = weight
        W.append(w)
        fidelity_scores.append(exp.score)

    W = np.vstack(W)
    mean_abs = np.mean(np.abs(W), axis=0)
    mean_signed = np.mean(W, axis=0)
    
    if noise_std > 0:
        mean_abs += np.random.normal(0.0, noise_std, size=n_features)
        if noise_clip_nonneg:
            mean_abs = np.clip(mean_abs, 0.0, None)
    
    return {
        "client": client_name,
        "n_client": n_client,
        "n_explained": len(W),
        "mean_abs": mean_abs,
        "mean_signed": mean_signed,
        "fidelity_scores": fidelity_scores,
    }


def compute_global_fidelity(model, X_test, y_test, global_mean_abs, device):
    model.eval()
    with torch.no_grad():
        logits = model(X_test).squeeze()
        y_prob_actual = torch.sigmoid(logits).cpu().numpy()
    
    X_np = X_test.detach().cpu().numpy()
    X_normalized = (X_np - X_np.mean(axis=0)) / (X_np.std(axis=0) + 1e-10)
    lime_predictions = np.dot(X_normalized, global_mean_abs)
    lime_predictions = (lime_predictions - lime_predictions.min()) / (lime_predictions.max() - lime_predictions.min() + 1e-10)
    
    return r2_score(y_prob_actual, lime_predictions)


def fedavg_aggregate_mean_abs(client_mean_abs, client_sizes):
    total = client_sizes.sum()
    weights = client_sizes / total if total > 0 else np.ones_like(client_sizes) / len(client_sizes)
    return (client_mean_abs * weights[:, None]).sum(axis=0)


def make_feature_ranking(column_names_list, global_mean_abs, top_k):
    idx = np.argsort(-global_mean_abs)[:top_k]
    return [(column_names_list[i], global_mean_abs[i]) for i in idx]


def make_bins(column_names_list, global_mean_abs, method, q_high, q_med, fixed_high, fixed_med):
    if method == "quantile":
        nonzero = global_mean_abs[global_mean_abs > 0]
        th_high = np.quantile(nonzero, q_high) if nonzero.size > 0 else 0.0
        th_med = np.quantile(nonzero, q_med) if nonzero.size > 0 else 0.0
    else:
        th_high = fixed_high
        th_med = fixed_med

    rows = []
    for name, val in zip(column_names_list, global_mean_abs):
        if val >= th_high:
            bin_label = "High"
        elif val >= th_med:
            bin_label = "Medium"
        else:
            bin_label = "Low"
        rows.append((name, val, bin_label))

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


def federated_lime_report(model, clients_data, column_names_list, num_instances, 
                         num_features_in_exp, num_samples, device, seed, 
                         noise_std, noise_clip_nonneg):
    summaries = []
    for client_name in clients_data.keys():
        summary = client_lime_summary(
            client_name, model, clients_data, column_names_list,
            num_instances, num_features_in_exp, num_samples, device, seed,
            noise_std, noise_clip_nonneg
        )
        summaries.append(summary)

    client_names = [s["client"] for s in summaries]
    client_mean_abs = np.vstack([s["mean_abs"] for s in summaries])
    client_sizes = np.array([s["n_client"] for s in summaries])
    client_fidelity = [np.mean(s["fidelity_scores"]) if s["fidelity_scores"] else 0.0 for s in summaries]

    global_mean_abs = fedavg_aggregate_mean_abs(client_mean_abs, client_sizes)

    if len(client_names) >= 2:
        sim = cosine_similarity(client_mean_abs)
    else:
        sim = np.ones((len(client_names), len(client_names)))

    return {
        "client_names": client_names,
        "client_sizes": client_sizes,
        "client_mean_abs": client_mean_abs,
        "global_mean_abs": global_mean_abs,
        "similarity_matrix": sim,
        "client_fidelity": client_fidelity,
    }


def save_lime_results(destination, lime_results, global_fidelity):
    client_names = lime_results["client_names"]
    client_sizes = lime_results["client_sizes"]
    client_mean_abs = lime_results["client_mean_abs"]
    global_mean_abs = lime_results["global_mean_abs"]
    sim = lime_results["similarity_matrix"]
    client_fidelity = lime_results["client_fidelity"]
    
    np.save(os.path.join(destination, "lime_global_mean_abs.npy"), global_mean_abs)
    np.save(os.path.join(destination, "lime_client_mean_abs.npy"), client_mean_abs)
    np.save(os.path.join(destination, "lime_similarity.npy"), sim)
    np.save(os.path.join(destination, "lime_client_names.npy"), np.array(client_names, dtype=object))
    np.save(os.path.join(destination, "lime_client_sizes.npy"), client_sizes)
    np.save(os.path.join(destination, "lime_client_fidelity.npy"), np.array(client_fidelity))
    np.save(os.path.join(destination, "lime_global_fidelity.npy"), np.array([global_fidelity]))


def print_fidelity_results(client_names, client_fidelity, global_fidelity):
    print("\n[FedLIME] Fidelity Scores (R² - how well LIME approximates the model):")
    print(f"  Global Fidelity: {global_fidelity:.4f}")
    print(f"  Per-Client Fidelity:")
    for cname, fid in zip(client_names, client_fidelity):
        print(f"    {cname}: {fid:.4f}")
    print(f"  Average Client Fidelity: {np.mean(client_fidelity):.4f}")


def main():
    global clients_data_global, input_dim_global, args_global, device_global
    global global_model, X_test_global, y_test_global
    
    parser = argparse.ArgumentParser(description="FedLIME with Flower Framework")
    
    parser.add_argument("--dataset_name", type=str, default="bank",
                       choices=["adult", "adult-age", "bank", "bank-age", "bank-age-5",
                               "law", "law-income", "default", "default-age", "acs"])
    parser.add_argument("--num_clients", type=int, default=3, choices=[3, 5, 10, 15])
    parser.add_argument("--epochs", type=int, default=2, help="Local epochs per round")
    parser.add_argument("--num_rounds", type=int, default=5, help="Number of FL rounds")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--fraction_fit", type=float, default=1.0,
                       help="Fraction of clients to sample for training")
    parser.add_argument("--fraction_evaluate", type=float, default=0.0,
                       help="Fraction of clients to sample for evaluation")
    parser.add_argument("--lime_action", type=str, default="all",
                       choices=["none", "aggregate", "noise", "rank", "bins", "all"],
                       help="LIME action to perform")
    parser.add_argument("--lime_instances", type=int, default=100,
                       help="Number of instances to explain per client")
    parser.add_argument("--lime_num_features", type=int, default=10,
                       help="Number of features in each LIME explanation")
    parser.add_argument("--lime_num_samples", type=int, default=2000,
                       help="Number of perturbation samples for LIME")
    parser.add_argument("--lime_noise_std", type=float, default=0.0, 
                       help="Std-dev of Gaussian noise for local DP")
    parser.add_argument("--lime_noise_clip", type=str, default="yes",
                       choices=["yes", "no"],
                       help="Clip client mean-abs weights to be non-negative after noise")
    parser.add_argument("--top_k", type=int, default=10, help="Top-K features for ranking")
    parser.add_argument("--bin_method", type=str, default="quantile",
                       choices=["quantile", "fixed"],
                       help="Method to create High/Medium/Low bins")
    parser.add_argument("--bin_high", type=float, default=0.66, help="Quantile for High")
    parser.add_argument("--bin_med", type=float, default=0.33, help="Quantile for Medium")
    parser.add_argument("--bin_fixed_high", type=float, default=0.01, help="Fixed threshold for High")
    parser.add_argument("--bin_fixed_med", type=float, default=0.005, help="Fixed threshold for Medium")
    parser.add_argument("--out_dir", type=str, default="flower_results", help="Output directory")
    
    args = parser.parse_args()
    args_global = args
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device("cpu")
    device_global = device
    
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
        raise ValueError("Dataset not supported")
    
    clients_data, X_test, y_test, sex_list, column_names_list, ytest_potential = load_dataset(
        url, dataset_name, args.num_clients, sensitive_feature
    )
    
    clients_data_global = clients_data
    X_test_global = X_test.to(device).float()
    y_test_global = y_test.to(device).float()
    
    input_dim = X_test.shape[1]
    input_dim_global = input_dim
    
    global_model = create_model(input_dim).to(device)
    
    print(f"\n{'='*60}")
    print(f"FedLIME with Flower Framework")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Clients: {len(clients_data)}")
    print(f"Rounds: {args.num_rounds}")
    print(f"Local epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"LIME action: {args.lime_action}")
    if args.lime_action != "none":
        print(f"LIME instances: {args.lime_instances}")
        if args.lime_noise_std > 0:
            print(f"Local DP noise std: {args.lime_noise_std}")
    print(f"{'='*60}\n")
    
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_evaluate,
        min_fit_clients=len(clients_data),
        min_evaluate_clients=0,
        min_available_clients=len(clients_data),
        evaluate_fn=evaluate_global_model,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=fl.common.ndarrays_to_parameters(
            [val.cpu().numpy() for _, val in global_model.state_dict().items()]
        ),
    )
    
    client_resources = {"num_cpus": 1, "num_gpus": 0}
    client_ids = list(clients_data.keys())
    
    print("Starting Flower simulation...\n")
    
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(clients_data),
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        client_resources=client_resources,
    )
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    
    acc_list = []
    bal_acc_list = []
    auprc_list = []
    
    if history.metrics_centralized:
        final_metrics = history.metrics_centralized
        
        if "accuracy" in final_metrics:
            acc_list = [m[1] for m in final_metrics["accuracy"]]
            final_acc = acc_list[-1]
            print(f"Final Accuracy: {final_acc:.4f}")
        
        if "balanced_accuracy" in final_metrics:
            bal_acc_list = [m[1] for m in final_metrics["balanced_accuracy"]]
            final_bal_acc = bal_acc_list[-1]
            print(f"Final Balanced Accuracy: {final_bal_acc:.4f}")
        
        if "auprc" in final_metrics:
            auprc_list = [m[1] for m in final_metrics["auprc"]]
            final_auprc = auprc_list[-1]
            print(f"Final AUPRC: {final_auprc:.4f}")
    
    print(f"{'='*60}\n")
    
    destination = f"./results/{dataset_name}/{args.out_dir}/"
    os.makedirs(destination, exist_ok=True)
    
    if acc_list:
        np.save(os.path.join(destination, "acc_curve.npy"), np.array(acc_list, dtype=np.float64))
    if bal_acc_list:
        np.save(os.path.join(destination, "bal_acc_curve.npy"), np.array(bal_acc_list, dtype=np.float64))
    if auprc_list:
        np.save(os.path.join(destination, "auprc_curve.npy"), np.array(auprc_list, dtype=np.float64))
    
    print(f"Training results saved to: {destination}")
    
    if args.lime_action == "none":
        print(f"\nDone. No LIME analysis requested.")
        return
    
    want_noise = (args.lime_action in ["noise", "all"]) or (args.lime_noise_std > 0 and args.lime_action in ["rank", "bins"])
    noise_std = float(args.lime_noise_std) if want_noise else 0.0
    noise_clip_nonneg = (args.lime_noise_clip == "yes")
    
    # Run LIME analysis
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
    
    global_fidelity = compute_global_fidelity(
        model=global_model,
        X_test=X_test_global,
        y_test=y_test_global,
        global_mean_abs=lime_results["global_mean_abs"],
        device=device,
    )
    
    if noise_std > 0:
        print(f"\n[FedLIME] Local DP noise: std={noise_std} (clip_nonneg={noise_clip_nonneg})")
    
    print_lime_summary(
        column_names=column_names_list,
        client_names=lime_results["client_names"],
        client_sizes=lime_results["client_sizes"],
        client_mean_abs=lime_results["client_mean_abs"],
        global_mean_abs=lime_results["global_mean_abs"],
        sim=lime_results["similarity_matrix"],
        top_k=args.top_k,
    )
    
    print_fidelity_results(
        lime_results["client_names"],
        lime_results["client_fidelity"],
        global_fidelity
    )
    
    save_lime_results(destination, lime_results, global_fidelity)
    
    if args.lime_action in ["rank", "all", "noise"]:
        ranking = make_feature_ranking(column_names_list, lime_results["global_mean_abs"], args.top_k)
        print(f"\n[FedLIME] Top {args.top_k} GLOBAL features (saved to CSV too):")
        for fname, val in ranking:
            print(f"  {fname}: {val:.6f}")
        
        with open(os.path.join(destination, "lime_feature_ranking.csv"), "w") as f:
            f.write("feature,mean_abs_weight\n")
            for fname, val in ranking:
                f.write(f"{fname},{val}\n")
    
    if args.lime_action in ["bins", "all"]:
        rows, th_high, th_med = make_bins(
            column_names_list, lime_results["global_mean_abs"],
            args.bin_method, args.bin_high, args.bin_med,
            args.bin_fixed_high, args.bin_fixed_med
        )
        print(f"\n[FedLIME] Binning thresholds -> High >= {th_high:.6f}, Medium >= {th_med:.6f} (method={args.bin_method})")
        print_bins(rows)
        
        with open(os.path.join(destination, "lime_feature_bins.csv"), "w") as f:
            f.write("feature,mean_abs_weight,bin\n")
            for feat, val, b in rows:
                f.write(f"{feat},{val},{b}\n")
    
    print(f"\nDone. All results saved to: {destination}")


if __name__ == "__main__":
    main()
