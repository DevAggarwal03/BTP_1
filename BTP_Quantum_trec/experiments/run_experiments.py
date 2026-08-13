"""
run_experiments.py
Single QuantumProtoNet end-to-end research entry point.

1. Loads TREC-50 data.
2. Runs QHBA for feature selection.
3. Episodically meta-trains QPN.
4. Evaluates on test episodes.
5. Generates visualisations.
"""
import sys
import os
import yaml
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
from quantum.evaluation.metrics import QuantumEvaluator
from quantum.evaluation.visualizer import QuantumSpaceVisualizer


def load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()

    print("=== Phase 1: Data Loading & Classical Preprocessing ===")
    loader = TRECLoader()
    dataset = loader.load_split(split='train')

    # Take a subset if needed, or use full dataset
    np.random.seed(42)
    # Using 1000 samples for a reasonable test run
    n_samples = min(1000, len(dataset['text']))
    indices = np.random.choice(len(dataset['text']), size=n_samples, replace=False)
    texts = [dataset['text'][i] for i in indices]
    labels = [dataset['label'][i] for i in indices]

    y = np.array(labels, dtype=int)
    class_names = [f"Class_{i}" for i in range(50)]

    preprocessor = TRECPreprocessor(
        encoder_model=cfg['data']['encoder_model'],
        n_features_lda=cfg['data']['n_features_lda'],
    )
    print(f"Running Classical Feature Extraction into {cfg['data']['n_features_lda']} dims...")
    X = preprocessor.fit_transform(texts, y)

    print("\n=== Phase 2: Quantum Honey Badger Feature Selection ===")
    qpn_trainer = QPNMasterTrainer(
        n_features=X.shape[1],
        epochs_per_eval=2,  # Quick eval for QHBA
        n_qubits=cfg['quantum']['n_qubits']
    )
    
    # Fast QHBA run for demo
    qpn_trainer.qhba.cfg = QHBAConfig(n_agents=5, max_iter=3, use_quantum_oracle=True)
    qhba_result = qpn_trainer.fit(X, y)

    best_features = qhba_result.selected_indices
    print(f"Best Feature Mask found by QHBA: {best_features}")
    
    # Filter dataset down to the best features
    X_selected = X[:, best_features]

    print("\n=== Phase 3: Episodic Meta-Training (QPN) ===")
    t_cfg = cfg['training']
    model = QuantumProtoNet(n_qubits=len(best_features))
    
    model = meta_train_qpn(
        model=model,
        X_train=X_selected,
        y_train=y,
        n_way=t_cfg['n_way'],
        k_shot=t_cfg['k_shot'],
        n_query=t_cfg['n_query'],
        n_train_episodes=10,  # Small number for demo
        lr=t_cfg['learning_rate'],
        lr_step_size=t_cfg['lr_step_size'],
        lr_gamma=t_cfg['lr_gamma'],
    )

    print("\n=== Phase 4: Final Evaluation ===")
    e_cfg = cfg['evaluation']
    
    test_pool = build_class_pool(X_selected, y)
    sampler = EpisodeSampler(test_pool, e_cfg['n_way'], e_cfg['k_shot'], e_cfg['n_query'])

    metrics = evaluate(model.predict, sampler, n_episodes=50)
    
    print("\nTest Results:")
    print(f"  Accuracy: {metrics['accuracy']['mean']:.2f}% ± {metrics['accuracy']['ci95']:.2f}%")
    print(f"  Weighted F1: {metrics['f1']['mean']:.2f}% ± {metrics['f1']['ci95']:.2f}%")

    print("\n=== Phase 5: Generating Visualizations ===")
    
    # We evaluate on one episode to get actual predictions and statevectors
    model.eval()
    ep = sampler.sample(1)[0]
    y_true = ep.query_y
    y_pred = model.predict(ep)
    
    evaluator = QuantumEvaluator(class_names=class_names)
    evaluator.plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png")
    
    # To plot statevectors, we need the actual states. 
    # For the t-SNE plot, we just pass some real output embeddings from the QPN
    import torch
    s_svs = model._encode_batch(ep.support_x, model.theta.detach().cpu().numpy())
    q_svs = model._encode_batch(ep.query_x, model.theta.detach().cpu().numpy())
    all_svs = s_svs + q_svs
    # Convert statevectors to concatenated real arrays
    sv_data = [np.concatenate((sv.data.real, sv.data.imag)) for sv in all_svs]
    all_y = np.concatenate((ep.support_y, ep.query_y))

    visualizer = QuantumSpaceVisualizer()
    visualizer.plot_tsne(sv_data, all_y, class_names, save_path="tsne_plot.png")
    
    print("Visualizations saved to confusion_matrix.png and tsne_plot.png")
    print("=== Pipeline Execution Complete ===")


if __name__ == "__main__":
    main()
