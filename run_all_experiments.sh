#!/bin/bash

# run_all_experiments.sh

set -e 

echo "FedLIME Experiments with Local Privacy"
echo ""
echo "Configuration:"
echo "  Communication Rounds: 50"
echo "  Local Epochs: 15"
echo "  Batch Size: 128"
echo "  Learning Rate: 0.001"
echo "  LIME Noise (σ): 0.01"
echo "  Seed: 0"
echo ""
echo "Total: 12 experiments (9 federated + 3 centralized)"
echo ""

START_TIME=$(date +%s)
EXPERIMENT_NUM=0

# Function to run experiment
run_experiment() {
    EXPERIMENT_NUM=$((EXPERIMENT_NUM + 1))
    echo ""
    echo "Experiment $EXPERIMENT_NUM/9: $1"
    echo "Command: $2"
    echo ""
    
    eval "$2"
    
    if [ $? -eq 0 ]; then
        echo "✓ Experiment $EXPERIMENT_NUM completed successfully"
    else
        echo "✗ Experiment $EXPERIMENT_NUM failed"
        exit 1
    fi
}

# Non-IID Settings (3 clients)

run_experiment "Adult-Age (Non-IID, 3 age groups)" \
"python fedlime_newaggmethod.py \
    --dataset_name adult-age \
    --num_clients 3 \
    --communication_rounds 50 \
    --epochs 15 \
    --batch_size 128 \
    --lr 0.001 \
    --use_pos_weight yes \
    --lime_action all \
    --lime_instances 100 \
    --lime_num_features 10 \
    --lime_num_samples 2000 \
    --lime_noise_std 0.01 \
    --lime_noise_clip yes \
    --out_dir noniid_adult_age_ldp \
    --seed 0"

run_experiment "Law-Income (Non-IID, 3 income groups)" \
"python fedlime_newaggmethod.py \
    --dataset_name law-income \
    --num_clients 3 \
    --communication_rounds 50 \
    --epochs 15 \
    --batch_size 128 \
    --lr 0.001 \
    --use_pos_weight yes \
    --lime_action all \
    --lime_instances 100 \
    --lime_num_features 10 \
    --lime_num_samples 2000 \
    --lime_noise_std 0.01 \
    --lime_noise_clip yes \
    --out_dir noniid_law_income_ldp \
    --seed 0"

run_experiment "Bank-Age (Non-IID, 3 age groups)" \
"python fedlime_newaggmethod.py \
    --dataset_name bank-age \
    --num_clients 3 \
    --communication_rounds 50 \
    --epochs 15 \
    --batch_size 128 \
    --lr 0.001 \
    --use_pos_weight yes \
    --lime_action all \
    --lime_instances 100 \
    --lime_num_features 10 \
    --lime_num_samples 2000 \
    --lime_noise_std 0.01 \
    --lime_noise_clip yes \
    --out_dir noniid_bank_age_ldp \
    --seed 0"

# IID Settings - 10 Clients

run_experiment "Bank (IID, 10 clients)" \
"python fedlime_newaggmethod.py \
    --dataset_name bank \
    --num_clients 10 \
    --communication_rounds 50 \
    --epochs 15 \
    --batch_size 128 \
    --lr 0.001 \
    --use_pos_weight yes \
    --lime_action all \
    --lime_instances 100 \
    --lime_num_features 10 \
    --lime_num_samples 2000 \
    --lime_noise_std 0.01 \
    --lime_noise_clip yes \
    --out_dir iid_bank_10clients_ldp \
    --seed 0"

run_experiment "Law (IID, 10 clients)" \
"python fedlime_newaggmethod.py \
    --dataset_name law \
    --num_clients 10 \
    --communication_rounds 50 \
    --epochs 15 \
    --batch_size 128 \
    --lr 0.001 \
    --use_pos_weight yes \
    --lime_action all \
    --lime_instances 100 \
    --lime_num_features 10 \
    --lime_num_samples 2000 \
    --lime_noise_std 0.01 \
    --lime_noise_clip yes \
    --out_dir iid_law_10clients_ldp \
    --seed 0"

run_experiment "Adult (IID, 10 clients)" \
"python fedlime_newaggmethod.py \
    --dataset_name adult \
    --num_clients 10 \
    --communication_rounds 50 \
    --epochs 15 \
    --batch_size 128 \
    --lr 0.001 \
    --use_pos_weight yes \
    --lime_action all \
    --lime_instances 100 \
    --lime_num_features 10 \
    --lime_num_samples 2000 \
    --lime_noise_std 0.01 \
    --lime_noise_clip yes \
    --out_dir iid_adult_10clients_ldp \
    --seed 0"

# IID Settings - 3 Clients

run_experiment "Bank (IID, 3 clients)" \
"python fedlime_newaggmethod.py \
    --dataset_name bank \
    --num_clients 3 \
    --communication_rounds 50 \
    --epochs 15 \
    --batch_size 128 \
    --lr 0.001 \
    --use_pos_weight yes \
    --lime_action all \
    --lime_instances 100 \
    --lime_num_features 10 \
    --lime_num_samples 2000 \
    --lime_noise_std 0.01 \
    --lime_noise_clip yes \
    --out_dir iid_bank_3clients_ldp \
    --seed 0"

run_experiment "Law (IID, 3 clients)" \
"python fedlime_newaggmethod.py \
    --dataset_name law \
    --num_clients 3 \
    --communication_rounds 50 \
    --epochs 15 \
    --batch_size 128 \
    --lr 0.001 \
    --use_pos_weight yes \
    --lime_action all \
    --lime_instances 100 \
    --lime_num_features 10 \
    --lime_num_samples 2000 \
    --lime_noise_std 0.01 \
    --lime_noise_clip yes \
    --out_dir iid_law_3clients_ldp \
    --seed 0"

run_experiment "Adult (IID, 3 clients)" \
"python fedlime_newaggmethod.py \
    --dataset_name adult \
    --num_clients 3 \
    --communication_rounds 50 \
    --epochs 15 \
    --batch_size 128 \
    --lr 0.001 \
    --use_pos_weight yes \
    --lime_action all \
    --lime_instances 100 \
    --lime_num_features 10 \
    --lime_num_samples 2000 \
    --lime_noise_std 0.01 \
    --lime_noise_clip yes \
    --out_dir iid_adult_3clients_ldp \
    --seed 0"

# Centralized (Global) Baselines

echo ""
echo "CENTRALIZED BASELINES"
echo ""

run_experiment "Adult (Centralized Baseline)" \
"python global_model_only.py \
    --dataset_name adult \
    --num_clients 10 \
    --epochs 750 \
    --batch_size 128 \
    --lr 0.001 \
    --use_pos_weight yes \
    --run_lime yes \
    --lime_instances 100 \
    --lime_num_features 10 \
    --lime_num_samples 2000"

run_experiment "Bank (Centralized Baseline)" \
"python global_model_only.py \
    --dataset_name bank \
    --num_clients 10 \
    --epochs 750 \
    --batch_size 128 \
    --lr 0.001 \
    --use_pos_weight yes \
    --run_lime yes \
    --lime_instances 100 \
    --lime_num_features 10 \
    --lime_num_samples 2000"

run_experiment "Law (Centralized Baseline)" \
"python global_model_only.py \
    --dataset_name law \
    --num_clients 10 \
    --epochs 750 \
    --batch_size 128 \
    --lr 0.001 \
    --use_pos_weight yes \
    --run_lime yes \
    --lime_instances 100 \
    --lime_num_features 10 \
    --lime_num_samples 2000"

# Generate Visualizations

echo ""
echo "GENERATING VISUALIZATIONS"
echo ""

echo "Visualizing federated experiments..."

# Non-IID
python visualize_lime_results.py --result_dir results/adult-age/noniid_adult_age_ldp --save
python visualize_lime_results.py --result_dir results/law-income/noniid_law_income_ldp --save
python visualize_lime_results.py --result_dir results/bank-age/noniid_bank_age_ldp --save

# IID 10 clients
python visualize_lime_results.py --result_dir results/bank/iid_bank_10clients_ldp --save
python visualize_lime_results.py --result_dir results/law/iid_law_10clients_ldp --save
python visualize_lime_results.py --result_dir results/adult/iid_adult_10clients_ldp --save

# IID 3 clients
python visualize_lime_results.py --result_dir results/bank/iid_bank_3clients_ldp --save
python visualize_lime_results.py --result_dir results/law/iid_law_3clients_ldp --save
python visualize_lime_results.py --result_dir results/adult/iid_adult_3clients_ldp --save

echo ""
echo "All visualizations generated"


END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "All Experiments Completed Successfully!"
echo ""
echo "Total experiments: 12 (9 federated + 3 centralized)"
echo "Total visualizations: 9 (federated only)"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "Federated results saved in:"
echo "  results/adult-age/noniid_adult_age_ldp/"
echo "  results/law-income/noniid_law_income_ldp/"
echo "  results/bank-age/noniid_bank_age_ldp/"
echo "  results/bank/iid_bank_10clients_ldp/"
echo "  results/law/iid_law_10clients_ldp/"
echo "  results/adult/iid_adult_10clients_ldp/"
echo "  results/bank/iid_bank_3clients_ldp/"
echo "  results/law/iid_law_3clients_ldp/"
echo "  results/adult/iid_adult_3clients_ldp/"
echo ""
echo "Centralized baselines saved in:"
echo "  results/adult/global_model_only/"
echo "  results/bank/global_model_only/"
echo "  results/law/global_model_only/"
echo ""
echo "Visualizations (plots/) created in each federated result directory"
echo ""
