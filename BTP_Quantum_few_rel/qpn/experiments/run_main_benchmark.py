# This script executes the main benchmark suite across the supported baselines and models.
# It orchestrates experiment runs and records outputs to the results directory.
import os
import time
import json
import numpy as np
from datetime import datetime

from qpn.config import STANDARD_SETTINGS, set_seed
from qpn.data import get_relation_pools
from qpn.episodes import EpisodeSampler
from qpn.eval.harness import evaluate
from qpn.eval.stats import paired_test

from qpn.baselines import (
    UntrainedProtoNet,
    TrainedProtoNet,
    ScikitLearnBaseline,
    QuantumKernelBaseline,
    QuantumKNNBaseline
)
from qpn.quantum.protonet import QuantumProtoNet
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

def run_main_benchmark(n_episodes=600, n_qubits=8, seed=42):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("results", timestamp)
    os.makedirs(out_dir, exist_ok=True)
    
    set_seed(seed)
    print(f"Starting Main Benchmark. Output dir: {out_dir}")
    
    train_pool, val_pool = get_relation_pools()
    
    results_db = {}
    
    print("Fitting QCHBA on train_pool (this might take a moment)...")
    from qpn.qchba import QCHBASelector
    from qpn.config import QCHBAConfig
    
    # We extract a subset of the train pool to quickly fit QCHBA
    train_X = []
    train_y = []
    for rel_idx, (rel, instances) in enumerate(train_pool.items()):
        # Use all instances for the most robust feature selection
        subset = instances
        train_X.extend(subset)
        train_y.extend([rel_idx] * len(subset))
            
    train_X = np.array(train_X)
    train_y = np.array(train_y)
    
    qchba_config = QCHBAConfig(n_features=n_qubits)
    qchba = QCHBASelector(qchba_config, pop_size=30, max_iter=100)
    qchba.fit(train_X, train_y)
    actual_indices = qchba.selected_indices
    
    # We define the models to test
    # Create the global preprocessors
    from qpn.quantum.encoding import QuantumFeaturePreprocessor
    from sklearn.preprocessing import MinMaxScaler
    
    # 1. Quantum Amplitude Preprocessor (pads to 2^n_qubits)
    qpn_preprocessor = QuantumFeaturePreprocessor(n_qubits, fm_kind="amplitude")
    qpn_preprocessor.fit(np.array(train_X), np.array(train_y), actual_indices)
    
    # 2. Classical Preprocessor (uses all QCHBA features, just minmax scaled)
    classical_scaler = MinMaxScaler()
    classical_scaler.fit(np.array(train_X)[:, actual_indices])
    
    # 3. QSVC Preprocessor (uses ZZ encoding, so it is choked to n_qubits features)
    qsvc_preprocessor = QuantumFeaturePreprocessor(n_qubits, fm_kind="zz")
    qsvc_preprocessor.fit(np.array(train_X), np.array(train_y), actual_indices)

    models = {
        "Classical ProtoNet (Untrained)": UntrainedProtoNet(),
        "Classical SVM (RBF)": ScikitLearnBaseline(SVC, kernel="rbf"),
        "Classical LogReg": ScikitLearnBaseline(LogisticRegression, max_iter=1000),
        "Classical kNN (k=1)": ScikitLearnBaseline(KNeighborsClassifier, n_neighbors=1),
        "Quantum QSVC (ZZ)": QuantumKernelBaseline(n_qubits=n_qubits, fm_kind="zz", qchba_indices=qsvc_preprocessor.selected_indices),
        "Quantum ProtoNet": QuantumProtoNet(n_qubits=n_qubits, fm_kind="amplitude", init_type="identity_block", cost_type="global")
    }
    
    for setting_name, config in STANDARD_SETTINGS.items():
        print(f"\n========================================")
        print(f"Running Benchmark: {setting_name}")
        print(f"========================================")
        results_db[setting_name] = {}
        
        # Test sampler (disjoint from train)
        test_sampler = EpisodeSampler(val_pool, config)
        
        for model_name, model in models.items():
            print(f"Evaluating {model_name}...")
            start_time = time.time()
            
            def model_fn(ep):
                from qpn.episodes import Episode
                
                if isinstance(model, QuantumProtoNet):
                    # Uses Amplitude Encoding (all features -> padded -> normalized)
                    s_x = qpn_preprocessor.transform(ep.support_x)
                    q_x = qpn_preprocessor.transform(ep.query_x)
                elif isinstance(model, QuantumKernelBaseline):
                    # Uses ZZ Encoding (funnel to n_qubits features)
                    s_x = qsvc_preprocessor.transform(ep.support_x)
                    q_x = qsvc_preprocessor.transform(ep.query_x)
                else:
                    # Classical Models (all features -> minmax scaled)
                    s_x = classical_scaler.transform(ep.support_x[:, actual_indices])
                    q_x = classical_scaler.transform(ep.query_x[:, actual_indices])
                
                matched_ep = Episode(
                    support_x=s_x,
                    support_y=ep.support_y,
                    query_x=q_x,
                    query_y=ep.query_y,
                    classes=ep.classes
                )
                
                if hasattr(model, 'predict'):
                    return model.predict(matched_ep)
                else:
                    return model(matched_ep)
            
            # -------------------------------------------------------------
            # NEW: Meta-train the Quantum model before evaluation
            # -------------------------------------------------------------
            if isinstance(model, QuantumProtoNet):
                print(f"  [Meta-Training] Optimizing {model_name}...")
                from qpn.eval.train import meta_train_qpn
                # Train thoroughly for publication readiness
                meta_train_qpn(model, train_pool, qpn_preprocessor, config, n_train_episodes=100, lr=0.05)
                
            metrics = evaluate(model_fn, test_sampler, n_episodes=n_episodes)
            elapsed = time.time() - start_time
            
            results_db[setting_name][model_name] = {
                "mean_acc": metrics["accuracy"]["mean"],
                "ci95_acc": metrics["accuracy"]["ci95"],
                "mean_f1": metrics["f1_weighted"]["mean"],
                "ci95_f1": metrics["f1_weighted"]["ci95"],
                "raw_accs": metrics["accuracy"]["raw"], # save for paired tests
                "time_s": elapsed
            }
            print(f"  Accuracy: {metrics['accuracy']['mean']:.2f} ± {metrics['accuracy']['ci95']:.2f}")
            
    # Write JSON
    with open(os.path.join(out_dir, "benchmark_results.json"), "w") as f:
        json.dump(results_db, f, indent=4)
        
    # Generate RESULTS.md
    generate_results_md(results_db, out_dir, n_episodes, n_qubits, seed)
    
def generate_results_md(results_db, out_dir, n_episodes, n_qubits, seed):
    md_path = os.path.join(out_dir, "RESULTS.md")
    # Also write to root directory as requested
    root_md_path = "RESULTS.md"
    
    content = f"# QPN Benchmark Results\n\n"
    content += f"**Episodes:** {n_episodes} | **Qubits:** {n_qubits} | **Seed:** {seed}\n\n"
    
    for setting, models in results_db.items():
        content += f"## {setting}\n\n"
        content += "| Model | Accuracy | Weighted F1 | Time (s) | Paired p-value (vs QPN) |\n"
        content += "|---|---|---|---|---|\n"
        
        qpn_accs = models["Quantum ProtoNet"]["raw_accs"]
        
        for name, data in models.items():
            acc_str = f"{data['mean_acc']:.2f} ± {data['ci95_acc']:.2f}"
            f1_str = f"{data['mean_f1']:.2f} ± {data['ci95_f1']:.2f}"
            time_str = f"{data['time_s']:.1f}"
            
            if name == "Quantum ProtoNet":
                p_val_str = "-"
            else:
                p_val, _ = paired_test(qpn_accs, data["raw_accs"])
                if p_val < 0.05:
                    p_val_str = f"**{p_val:.3f}**"
                else:
                    p_val_str = f"{p_val:.3f}"
                    
            content += f"| {name} | {acc_str} | {f1_str} | {time_str} | {p_val_str} |\n"
            
        content += "\n"
        
    with open(md_path, "w") as f:
        f.write(content)
        
    with open(root_md_path, "w") as f:
        f.write(content)
        
    print(f"Wrote RESULTS.md to {root_md_path}")

if __name__ == "__main__":
    # Use smaller n_episodes for quick local run, set to 600 or 2000 for actual paper
    run_main_benchmark(n_episodes=50, n_qubits=4)
