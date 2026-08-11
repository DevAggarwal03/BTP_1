# This script runs ablation studies for the prototype network and its variants.
# It sweeps configuration choices and saves the resulting metrics for analysis.
import os
import time
import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from qpn.config import STANDARD_SETTINGS, set_seed
from qpn.data import get_relation_pools
from qpn.episodes import EpisodeSampler
from qpn.eval.harness import evaluate
from qpn.quantum.protonet import QuantumProtoNet

def run_ablations(n_episodes=100, seed=42):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("results", f"ablations_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    
    set_seed(seed)
    print(f"Starting Ablations. Output dir: {out_dir}")
    
    train_pool, val_pool = get_relation_pools()
        
    config = STANDARD_SETTINGS["5w1s"]
    test_sampler = EpisodeSampler(val_pool, config)
    
    print("  Fitting QCHBA on train_pool (this might take a moment)...")
    from qpn.qchba import QCHBASelector
    from qpn.config import QCHBAConfig
    
    # We extract a subset of the train pool to quickly fit QCHBA for all ablations
    train_X = []
    train_y = []
    for rel_idx, (rel, instances) in enumerate(train_pool.items()):
        # Use all instances for the most robust feature selection
        subset = instances
        train_X.extend(subset)
        train_y.extend([rel_idx] * len(subset))
            
    train_X = np.array(train_X)
    train_y = np.array(train_y)
    
    qchba_config = QCHBAConfig(n_features=4) # Using 4 as baseline for selector/encoding ablations
    qchba = QCHBASelector(qchba_config, pop_size=30, max_iter=100)
    qchba.fit(train_X, train_y)
    actual_indices_4q = qchba.selected_indices
    
    from qpn.eval.train import meta_train_qpn
    from qpn.quantum.encoding import QuantumFeaturePreprocessor
    from qpn.episodes import Episode

    # Create 4-qubit global preprocessor for the static ablations
    prep_4q = QuantumFeaturePreprocessor(4)
    prep_4q.fit(train_X, train_y, actual_indices_4q)
    
    def transform_ep(ep, prep):
        s_x = prep.transform(ep.support_x)
        q_x = prep.transform(ep.query_x)
        return Episode(s_x, ep.support_y, q_x, ep.query_y, ep.classes)

    # 1. Encoding Ablation (Angle vs ZZ vs Amplitude) at 4 qubits
    print("\n--- Ablation: Encoding ---")
    encoding_results = {}
    for fm in ["angle", "zz", "amplitude"]:
        model = QuantumProtoNet(n_qubits=4, fm_kind=fm, init_type="identity_block", cost_type="global")
        # Since prep_4q was built with zz, we need a specific preprocessor for amplitude
        if fm == "amplitude":
            prep = QuantumFeaturePreprocessor(4, fm_kind="amplitude")
            prep.fit(train_X, train_y, actual_indices_4q)
        else:
            prep = prep_4q
            
        model = meta_train_qpn(model, train_pool, prep, config, n_train_episodes=100)
        def model_fn(ep, p=prep): return model.predict(transform_ep(ep, p))
        metrics = evaluate(model_fn, test_sampler, n_episodes=n_episodes)
        encoding_results[fm] = metrics["accuracy"]["mean"]
        print(f"  {fm}: {metrics['accuracy']['mean']:.2f}")
        
    # 2. Qubit Count Sweep (2, 4, 6, 8) with ZZ
    print("\n--- Ablation: Qubit Count ---")
    qubit_results = {}
    qubits = [2, 4, 6, 8]
    for nq in qubits:
        temp_qchba_config = QCHBAConfig(n_features=nq)
        temp_qchba = QCHBASelector(temp_qchba_config, pop_size=30, max_iter=100)
        temp_qchba.fit(train_X, train_y)
        
        prep_nq = QuantumFeaturePreprocessor(nq)
        prep_nq.fit(train_X, train_y, temp_qchba.selected_indices)
        
        model = QuantumProtoNet(n_qubits=nq, fm_kind="zz", init_type="identity_block", cost_type="global")
        model = meta_train_qpn(model, train_pool, prep_nq, config, n_train_episodes=100)
        def model_fn(ep, p=prep_nq): return model.predict(transform_ep(ep, p))
        metrics = evaluate(model_fn, test_sampler, n_episodes=n_episodes)
        qubit_results[nq] = metrics["accuracy"]["mean"]
        print(f"  n={nq}: {metrics['accuracy']['mean']:.2f}")
        
    # 3. Selector Ablation
    print("\n--- Ablation: Selector ---")
    selector_results = {}
    
    # ANOVA-F Baseline
    prep_anova = QuantumFeaturePreprocessor(4)
    prep_anova.fit(train_X, train_y, None) # None triggers ANOVA-F internally in fit()
    
    model_anova = QuantumProtoNet(n_qubits=4, fm_kind="zz", init_type="identity_block", cost_type="global")
    model_anova = meta_train_qpn(model_anova, train_pool, prep_anova, config, n_train_episodes=100)
    def model_fn_anova(ep): return model_anova.predict(transform_ep(ep, prep_anova))
    metrics_anova = evaluate(model_fn_anova, test_sampler, n_episodes=n_episodes)
    selector_results["ANOVA-F"] = metrics_anova["accuracy"]["mean"]
    print(f"  ANOVA-F: {metrics_anova['accuracy']['mean']:.2f}")

    # Actual QCHBA
    model_qchba = QuantumProtoNet(n_qubits=4, fm_kind="zz", init_type="identity_block", cost_type="global")
    model_qchba = meta_train_qpn(model_qchba, train_pool, prep_4q, config, n_train_episodes=100)
    def model_fn_qchba(ep): return model_qchba.predict(transform_ep(ep, prep_4q))
    metrics_qchba = evaluate(model_fn_qchba, test_sampler, n_episodes=n_episodes)
    selector_results["QCHBA"] = metrics_qchba["accuracy"]["mean"]
    print(f"  QCHBA: {metrics_qchba['accuracy']['mean']:.2f}")
    
    # 4. Circuit Depth Ablation (Ansatz Reps)
    print("\n--- Ablation: Circuit Depth (Ansatz Reps) ---")
    depth_results = {}
    for reps in [1, 2, 3, 4]:
        model = QuantumProtoNet(n_qubits=4, fm_kind="zz", ansatz_reps=reps, init_type="identity_block", cost_type="global")
        model = meta_train_qpn(model, train_pool, prep_4q, config, n_train_episodes=100)
        def model_fn(ep): return model.predict(transform_ep(ep, prep_4q))
        metrics = evaluate(model_fn, test_sampler, n_episodes=n_episodes)
        depth_results[reps] = metrics["accuracy"]["mean"]
        print(f"  Reps={reps}: {metrics['accuracy']['mean']:.2f}")
        
    # 5. Cost Function Ablation
    print("\n--- Ablation: Cost Function ---")
    cost_results = {}
    for ctype in ["global", "local"]:
        model = QuantumProtoNet(n_qubits=4, fm_kind="zz", init_type="identity_block", cost_type=ctype)
        model = meta_train_qpn(model, train_pool, prep_4q, config, n_train_episodes=100)
        def model_fn(ep): return model.predict(transform_ep(ep, prep_4q))
        metrics = evaluate(model_fn, test_sampler, n_episodes=n_episodes)
        cost_results[ctype] = metrics["accuracy"]["mean"]
        print(f"  {ctype}: {metrics['accuracy']['mean']:.2f}")
        
    results = {
        "encoding": encoding_results,
        "qubits": qubit_results,
        "selector": selector_results,
        "depth": depth_results,
        "cost": cost_results
    }
    
    with open(os.path.join(out_dir, "ablations.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    # Plot qubit sweep
    plt.figure()
    plt.plot(qubits, [qubit_results[q] for q in qubits], 'o-')
    plt.xlabel('Number of Qubits')
    plt.ylabel('Accuracy (5w1s)')
    plt.title('Accuracy vs Number of Qubits')
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "qubits_sweep.png"))
    
    # Append to RESULTS.md
    md_content = "\n## Ablations (5-way 1-shot)\n\n"
    
    md_content += "### 1. Feature Selector\n"
    md_content += "| Selector | Accuracy |\n|---|---|\n"
    for sel, acc in selector_results.items():
        md_content += f"| {sel} | {acc:.2f} |\n"
        
    md_content += "\n### 2. Encoding Map\n"
    md_content += "| Encoding | Accuracy |\n|---|---|\n"
    for enc, acc in encoding_results.items():
        md_content += f"| {enc} | {acc:.2f} |\n"
        
    md_content += "\n### 3. Qubit Count Sweep\n"
    md_content += "| Qubits | Accuracy |\n|---|---|\n"
    for nq, acc in qubit_results.items():
        md_content += f"| {nq} | {acc:.2f} |\n"
        
    md_content += "\n### 4. Circuit Depth (Ansatz Reps)\n"
    md_content += "| Reps | Accuracy |\n|---|---|\n"
    for reps, acc in depth_results.items():
        md_content += f"| {reps} | {acc:.2f} |\n"
        
    md_content += "\n### 5. Cost Function (Global vs Local)\n"
    md_content += "| Cost Type | Accuracy |\n|---|---|\n"
    for ctype, acc in cost_results.items():
        md_content += f"| {ctype} | {acc:.2f} |\n"
        
    with open("RESULTS.md", "a") as f:
        f.write(md_content)
    
    print(f"\nDone. Results in {out_dir}")

if __name__ == "__main__":
    run_ablations(n_episodes=50)
