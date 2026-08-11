import os
from qpn.experiments.run_main_benchmark import run_main_benchmark
from qpn.experiments.run_ablations import run_ablations

def main():
    print("=========================================================")
    print("  Starting Full QPN Benchmark Suite (Thesis Run)")
    print("=========================================================")
    
    # Fast testing settings (change back to 600 for full paper)
    n_episodes_benchmark = 20
    n_episodes_ablations = 20
    n_qubits = 9 # Amplitude encoding for ~333 features (2^9 = 512)
    
    # 1. Clean up old RESULTS.md if it exists
    if os.path.exists("RESULTS.md"):
        print("Removing old RESULTS.md...")
        os.remove("RESULTS.md")
        
    # 2. Run Main Benchmark across all 4 settings
    print("\n>>> Phase 1: Main Benchmarks (5w1s, 5w5s, 10w1s, 10w5s)")
    run_main_benchmark(n_episodes=n_episodes_benchmark, n_qubits=n_qubits, seed=42)
    
    # 3. Run Ablations
    print("\n>>> Phase 2: Ablation Studies")
    run_ablations(n_episodes=n_episodes_ablations, seed=42)
    
    print("\n=========================================================")
    print("  All experiments complete! Full report in RESULTS.md")
    print("=========================================================")

if __name__ == "__main__":
    main()
