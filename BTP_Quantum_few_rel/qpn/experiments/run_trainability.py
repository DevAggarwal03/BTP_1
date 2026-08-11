# This script evaluates whether the proposed approach remains trainable under different settings.
# It gathers results from repeated runs and summarizes them for reporting.
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from qpn.eval.trainability import estimate_gradient_variance, estimate_kernel_concentration

def run_trainability_experiments():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("results", timestamp)
    os.makedirs(out_dir, exist_ok=True)
    
    qubit_range = [2, 4, 6, 8, 10, 12]
    depth_range = [1, 2, 4, 6]
    
    results = {
        "global_var_qubits": [],
        "local_var_qubits": [],
        "global_var_depth": [],
        "local_var_depth": [],
        "kernel_concentration": []
    }
    
    print(f"Starting Trainability Experiments. Output dir: {out_dir}")
    
    # 1. Gradient variance vs n_qubits (fixed depth L=2)
    fixed_depth = 2
    for nq in qubit_range:
        print(f"Estimating Var vs Qubits (n={nq}, L={fixed_depth})...")
        var_g = estimate_gradient_variance(n_qubits=nq, depth=fixed_depth, cost_type="global", n_samples=100)
        var_l = estimate_gradient_variance(n_qubits=nq, depth=fixed_depth, cost_type="local", n_samples=100)
        
        results["global_var_qubits"].append(var_g)
        results["local_var_qubits"].append(var_l)
        
        print(f"  Global Var: {var_g:.2e} | Local Var: {var_l:.2e}")
        
    # 2. Gradient variance vs depth (fixed n_qubits=8)
    fixed_qubits = 8
    for d in depth_range:
        print(f"Estimating Var vs Depth (n={fixed_qubits}, L={d})...")
        var_g = estimate_gradient_variance(n_qubits=fixed_qubits, depth=d, cost_type="global", n_samples=100)
        var_l = estimate_gradient_variance(n_qubits=fixed_qubits, depth=d, cost_type="local", n_samples=100)
        
        results["global_var_depth"].append(var_g)
        results["local_var_depth"].append(var_l)
        
        print(f"  Global Var: {var_g:.2e} | Local Var: {var_l:.2e}")
        
    # 3. Kernel Concentration vs n_qubits
    for nq in qubit_range:
        print(f"Estimating Kernel Concentration (n={nq})...")
        var_k = estimate_kernel_concentration(n_qubits=nq, n_samples=100)
        results["kernel_concentration"].append(var_k)
        print(f"  Kernel Var: {var_k:.2e}")
        
    # Save raw results
    with open(os.path.join(out_dir, "trainability_results.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    # Plot 1: Variance vs n_qubits
    plt.figure(figsize=(8, 6))
    plt.plot(qubit_range, results["global_var_qubits"], 'o-', label='Global Cost (Exponential Decay)')
    plt.plot(qubit_range, results["local_var_qubits"], 's-', label='Local Cost (Polynomial Decay)')
    plt.yscale('log')
    plt.xlabel('Number of Qubits ($n$)')
    plt.ylabel('Variance of Gradient $\\text{Var}[\\partial C / \\partial \\theta_k]$')
    plt.title(f'Gradient Variance vs Qubits (Depth L={fixed_depth})')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "variance_vs_qubits.png"), dpi=300)
    
    # Plot 2: Variance vs Depth
    plt.figure(figsize=(8, 6))
    plt.plot(depth_range, results["global_var_depth"], 'o-', label='Global Cost')
    plt.plot(depth_range, results["local_var_depth"], 's-', label='Local Cost')
    plt.yscale('log')
    plt.xlabel('Ansatz Depth ($L$)')
    plt.ylabel('Variance of Gradient $\\text{Var}[\\partial C / \\partial \\theta_k]$')
    plt.title(f'Gradient Variance vs Depth (Qubits $n$={fixed_qubits})')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "variance_vs_depth.png"), dpi=300)
    
    # Plot 3: Kernel Concentration
    plt.figure(figsize=(8, 6))
    plt.plot(qubit_range, results["kernel_concentration"], '^-', color='green')
    plt.yscale('log')
    plt.xlabel('Number of Qubits ($n$)')
    plt.ylabel('Variance of Off-Diagonal Kernel Elements')
    plt.title('Kernel Concentration vs Qubits')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "kernel_concentration.png"), dpi=300)
    
    print(f"Experiments complete. Artifacts saved to {out_dir}")

if __name__ == "__main__":
    run_trainability_experiments()
