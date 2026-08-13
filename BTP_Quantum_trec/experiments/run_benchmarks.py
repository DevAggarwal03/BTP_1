"""
run_benchmarks.py
Full head-to-head benchmark across 6 models and 4 settings.
Generates RESULTS.md with paired Wilcoxon significance tests.
"""
import sys
import os
import yaml
import time
import json
from datetime import datetime
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.trec_loader import TRECLoader
from data.preprocessor import TRECPreprocessor
from data.episode_sampler import build_class_pool, EpisodeSampler
from quantum.feature_selection.qhba import QHBAConfig
from quantum.training.outer_loop import QPNMasterTrainer
from quantum.training.qpn_model import QuantumProtoNet
from quantum.eval.train import meta_train_qpn
from quantum.eval.harness import evaluate
from quantum.eval.stats import paired_test

from baselines import (
    UntrainedProtoNet,
    TrainedProtoNet,
    get_classical_ml_baselines,
    QuantumKernelBaseline
)


def load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", f"benchmark_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    
    print("=== Phase 1: Data Loading & Preprocessing ===")
    loader = TRECLoader()
    dataset = loader.load_split(split='train')
    
    np.random.seed(42)
    n_samples = min(2500, len(dataset['text']))
    indices = np.random.choice(len(dataset['text']), size=n_samples, replace=False)
    texts = [dataset['text'][i] for i in indices]
    y = np.array([dataset['label'][i] for i in indices], dtype=int)

    preprocessor = TRECPreprocessor(
        encoder_model=cfg['data']['encoder_model'],
        n_features_lda=cfg['data']['n_features_lda'],
    )
    X = preprocessor.fit_transform(texts, y)
    
    print("\n=== Phase 2: Quantum Honey Badger Feature Selection ===")
    qpn_trainer = QPNMasterTrainer(
        n_features=X.shape[1],
        epochs_per_eval=2,
        n_qubits=cfg['quantum']['n_qubits']
    )
    qpn_trainer.qhba.cfg = QHBAConfig(n_agents=10, max_iter=5, use_quantum_oracle=True)
    qhba_result = qpn_trainer.fit(X, y)
    best_features = qhba_result.selected_indices
    X_selected = X[:, best_features]
    print(f"Features selected for ALL models: {best_features}")
    
    # We create a single pool that we'll sample from. 
    pool = build_class_pool(X_selected, y)

    # 4 Settings
    settings = [
        {"name": "5w1s", "n_way": 5, "k_shot": 1, "n_query": 15},
        {"name": "5w5s", "n_way": 5, "k_shot": 5, "n_query": 15},
        {"name": "10w1s", "n_way": 10, "k_shot": 1, "n_query": 15},
        {"name": "10w5s", "n_way": 10, "k_shot": 5, "n_query": 15},
    ]

    # Models dictionary builder
    def get_models(n_features):
        models = {}
        models["Classical ProtoNet (Untrained)"] = UntrainedProtoNet()
        models["Classical ProtoNet (Trained MLP)"] = TrainedProtoNet(input_dim=n_features)
        
        for name, baseline in get_classical_ml_baselines().items():
            models[f"Classical {name}"] = baseline
            
        models["Quantum QSVC (Angle)"] = QuantumKernelBaseline(n_qubits=n_features)
        models["Quantum ProtoNet (Ours)"] = QuantumProtoNet(n_qubits=n_features)
        return models

    results_data = {}
    
    for setting in settings:
        s_name = setting["name"]
        print(f"\n=== Running Setting: {s_name} ===")
        
        try:
            sampler = EpisodeSampler(pool, setting["n_way"], setting["k_shot"], setting["n_query"])
        except ValueError as e:
            print(f"Skipping {s_name} due to lack of eligible classes: {e}")
            continue

        models = get_models(len(best_features))
        
        # Meta-train the trained models
        print("  Meta-training Classical ProtoNet...")
        models["Classical ProtoNet (Trained MLP)"].train_epoch(sampler, n_episodes=50)
        
        print("  Meta-training Quantum ProtoNet (Ours)...")
        models["Quantum ProtoNet (Ours)"] = meta_train_qpn(
            model=models["Quantum ProtoNet (Ours)"],
            X_train=X_selected,
            y_train=y,
            n_way=setting["n_way"],
            k_shot=setting["k_shot"],
            n_query=setting["n_query"],
            n_train_episodes=50,
            lr=cfg['training']['learning_rate']
        )
        
        setting_results = {}
        # We need the QPN raw scores for paired testing
        print("  Evaluating QPN...")
        qpn_metrics = evaluate(models["Quantum ProtoNet (Ours)"].predict, sampler, n_episodes=50)
        qpn_raw_acc = qpn_metrics["accuracy"]["raw"]
        
        for name, model in models.items():
            print(f"  Evaluating {name}...")
            start_t = time.time()
            if name == "Quantum ProtoNet (Ours)":
                metrics = qpn_metrics
            else:
                metrics = evaluate(model, sampler, n_episodes=50)
            elapsed = time.time() - start_t
            
            p_val, _ = paired_test(qpn_raw_acc, metrics["accuracy"]["raw"])
            
            setting_results[name] = {
                "accuracy": metrics["accuracy"]["mean"],
                "acc_ci": metrics["accuracy"]["ci95"],
                "f1": metrics["f1"]["mean"],
                "f1_ci": metrics["f1"]["ci95"],
                "time": elapsed,
                "p_value": p_val if name != "Quantum ProtoNet (Ours)" else "-"
            }
            
        results_data[s_name] = setting_results

    # Save JSON
    with open(os.path.join(out_dir, "benchmark_results.json"), "w") as f:
        json.dump(results_data, f, indent=4)
        
    # Generate Markdown
    md_path = os.path.join(out_dir, "RESULTS.md")
    with open(md_path, "w") as f:
        f.write("# TREC-50 Head-to-Head Benchmark Results\n\n")
        
        for s_name, data in results_data.items():
            f.write(f"### Setting: {s_name}\n\n")
            f.write("| Model | Accuracy | Weighted F1 | Time (s) | p-value (vs QPN) |\n")
            f.write("|---|---|---|---|---|\n")
            
            for name, res in data.items():
                acc = f"{res['accuracy']:.1f} ± {res['acc_ci']:.1f}"
                f1 = f"{res['f1']:.1f} ± {res['f1_ci']:.1f}"
                t = f"{res['time']:.1f}"
                pval = res['p_value']
                
                if name == "Quantum ProtoNet (Ours)":
                    f.write(f"| **{name}** | **{acc}** | **{f1}** | {t} | — |\n")
                else:
                    pv_str = f"{pval:.4f}" if pval != "-" else "-"
                    if pval != "-" and pval < 0.05:
                        pv_str = f"**{pv_str}**"
                    f.write(f"| {name} | {acc} | {f1} | {t} | {pv_str} |\n")
            
            f.write("\n")
            
    print(f"\n=== Benchmark Complete. Results saved to {out_dir}/RESULTS.md ===")

if __name__ == "__main__":
    main()
