import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_lime_results(result_dir):
    data = {}
    
    data['global_mean_abs'] = np.load(os.path.join(result_dir, "lime_global_mean_abs.npy"))
    data['client_mean_abs'] = np.load(os.path.join(result_dir, "lime_client_mean_abs.npy"))
    data['similarity'] = np.load(os.path.join(result_dir, "lime_similarity.npy"))
    data['client_names'] = np.load(os.path.join(result_dir, "lime_client_names.npy"), allow_pickle=True)
    data['client_sizes'] = np.load(os.path.join(result_dir, "lime_client_sizes.npy"))
    
    try:
        data['client_fidelity'] = np.load(os.path.join(result_dir, "lime_client_fidelity.npy"))
        data['global_fidelity'] = np.load(os.path.join(result_dir, "lime_global_fidelity.npy"))[0]
    except:
        data['client_fidelity'] = None
        data['global_fidelity'] = None
    
    n_features = len(data['global_mean_abs'])
    
    try:
        import pandas as pd
        
        bins_csv = os.path.join(result_dir, "lime_feature_bins.csv")
        if os.path.exists(bins_csv):
            df = pd.read_csv(bins_csv)
            data['feature_names'] = df['feature'].values.tolist()
        else:
            ranking_csv = os.path.join(result_dir, "lime_feature_ranking.csv")
            if os.path.exists(ranking_csv):
                df = pd.read_csv(ranking_csv)
                feature_names_from_csv = df['feature'].values.tolist()
                if len(feature_names_from_csv) < n_features:
                    for i in range(len(feature_names_from_csv), n_features):
                        feature_names_from_csv.append(f"feature_{i}")
                data['feature_names'] = feature_names_from_csv[:n_features]
            else:
                data['feature_names'] = [f"feature_{i}" for i in range(n_features)]
    except Exception as e:
        data['feature_names'] = [f"feature_{i}" for i in range(n_features)]
    
    if len(data['feature_names']) != n_features:
        print(f"Warning: Feature names mismatch. Expected {n_features}, got {len(data['feature_names'])}. Using generic names.")
        data['feature_names'] = [f"feature_{i}" for i in range(n_features)]
    
    return data


def plot_global_feature_importance(data, save_path=None):
    global_importance = data['global_mean_abs']
    feature_names = data['feature_names']
    
    sorted_idx = np.argsort(-global_importance)
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_values = global_importance[sorted_idx]
    
    plt.figure(figsize=(10, max(6, len(sorted_features) * 0.3)))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_values)))
    plt.barh(range(len(sorted_features)), sorted_values, color=colors)
    plt.yticks(range(len(sorted_features)), sorted_features)
    plt.xlabel('Mean Absolute LIME Weight', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('Global Feature Importance (All Features)', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close() 


def plot_similarity_matrix(data, save_path=None):
    similarity = data['similarity']
    client_names = data['client_names']
    
    plt.figure(figsize=(8, 7))
    sns.heatmap(similarity, 
                xticklabels=client_names, 
                yticklabels=client_names,
                cmap='coolwarm', 
                annot=True, 
                fmt='.3f',
                vmin=-1, 
                vmax=1,
                center=0,
                cbar_kws={'label': 'Cosine Similarity'})
    plt.title('Client Explanation Similarity (Cosine)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_fidelity_scores(data, save_path=None):
    client_names = data['client_names']
    client_fidelity = data['client_fidelity']
    global_fidelity = data['global_fidelity']
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    colors = ['green' if f > 0.7 else 'orange' if f > 0.5 else 'red' for f in client_fidelity]
    bars = ax.bar(range(len(client_names)), client_fidelity, color=colors, alpha=0.7, label='Client Fidelity')
    
    if global_fidelity is not None:
        ax.axhline(y=global_fidelity, color='blue', linestyle='--', linewidth=2, 
                   label=f'Global Fidelity: {global_fidelity:.3f}')
    
    avg_fidelity = np.mean(client_fidelity)
    ax.axhline(y=avg_fidelity, color='purple', linestyle=':', linewidth=2, 
               label=f'Avg Client Fidelity: {avg_fidelity:.3f}')
    
    ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.3, linewidth=1)
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.3, linewidth=1)
    
    ax.set_xticks(range(len(client_names)))
    ax.set_xticklabels(client_names, rotation=45, ha='right')
    ax.set_ylabel('R² Score (Fidelity)', fontsize=12)
    ax.set_xlabel('Client', fontsize=12)
    ax.set_title('LIME Fidelity Scores (How Well LIME Explains the Model)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.legend(loc='best')
    ax.grid(axis='y', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars, client_fidelity)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close() 
    client_names = data['client_names']
    client_sizes = data['client_sizes']
    client_fidelity = data['client_fidelity']
    
    if client_fidelity is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Client sizes
        axes[0].bar(range(len(client_names)), client_sizes, color='steelblue', alpha=0.7)
        axes[0].set_xticks(range(len(client_names)))
        axes[0].set_xticklabels(client_names, rotation=45, ha='right')
        axes[0].set_ylabel('Number of Samples', fontsize=11)
        axes[0].set_title('Client Dataset Sizes', fontsize=12, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Client fidelity
        colors = ['green' if f > 0.7 else 'orange' if f > 0.5 else 'red' for f in client_fidelity]
        axes[1].bar(range(len(client_names)), client_fidelity, color=colors, alpha=0.7)
        axes[1].axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Good (>0.7)')
        axes[1].axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Fair (>0.5)')
        axes[1].set_xticks(range(len(client_names)))
        axes[1].set_xticklabels(client_names, rotation=45, ha='right')
        axes[1].set_ylabel('R² Score', fontsize=11)
        axes[1].set_title('Client LIME Fidelity', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
        
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.bar(range(len(client_names)), client_sizes, color='steelblue', alpha=0.7)
        ax.set_xticks(range(len(client_names)))
        ax.set_xticklabels(client_names, rotation=45, ha='right')
        ax.set_ylabel('Number of Samples', fontsize=11)
        ax.set_title('Client Dataset Sizes', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close() 


def main():
    parser = argparse.ArgumentParser(description="Visualize FedLIME results")
    parser.add_argument("--result_dir", type=str, required=True, help="Path to results directory")
    parser.add_argument("--save", action="store_true", help="Save plots to files")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save plots (default: result_dir/plots)")
    
    args = parser.parse_args()
    
    print(f"Loading results from: {args.result_dir}")
    data = load_lime_results(args.result_dir)
    print(f"Loaded: {len(data['client_names'])} clients, {len(data['feature_names'])} features")
    
    if args.save:
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = os.path.join(args.result_dir, "plots")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving plots to: {output_dir}")
    else:
        output_dir = None
    
    print("\n1. Global Feature Importance (All Features)...")
    plot_global_feature_importance(data, 
                                   save_path=os.path.join(output_dir, "feature_importance.png") if output_dir else None)
    
    print("2. Cosine Similarity Matrix...")
    plot_similarity_matrix(data, 
                          save_path=os.path.join(output_dir, "cosine_similarity.png") if output_dir else None)
    
    print("3. Fidelity Scores...")
    plot_fidelity_scores(data,
                        save_path=os.path.join(output_dir, "fidelity_scores.png") if output_dir else None)
    
    print("\nVisualization complete!")
    print(f"\nGenerated 3 plots:")
    print(f"  1. feature_importance.png - All features ranked by importance")
    print(f"  2. cosine_similarity.png - Client explanation similarity")
    print(f"  3. fidelity_scores.png - How well LIME explains the model")


if __name__ == "__main__":
    main()
