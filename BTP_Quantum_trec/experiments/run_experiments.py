"""
run_experiments.py
End-to-end QPN research pipeline for TREC-50.

Architecture aligned with BTP_Quantum_few_rel/qpn/experiments/run_main_benchmark.py:

1. Loads TREC-50 data.
2. Classical feature extraction (SBERT -> MinMax -> LDA -> 32 dims).
3. QHBA quantum feature selection (selects best n_qubits features).
4. QuantumFeaturePreprocessor fit (pads + L2-normalizes for amplitude encoding).
5. Episodic meta-training with global_preprocessor (matching few_rel's train.py).
6. Full evaluation with t-CI progress printing (matching few_rel's harness.py).
7. Generates visualizations.
"""
import sys
import os
import yaml
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.trec_loader import TRECLoader
from data.preprocessor import TRECPreprocessor, QuantumFeaturePreprocessor
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
    q_cfg  = cfg['quantum']
    t_cfg  = cfg['training']
    e_cfg  = cfg['evaluation']
    qh_cfg = cfg['qhba']

    n_qubits  = q_cfg['n_qubits']       # 8
    cost_type = q_cfg['cost_type']       # 'global'
    init_type = q_cfg['init_type']       # 'identity_block'

    # ---------------------------------------------------------------
    print("=== Phase 1: Data Loading & Classical Preprocessing ===")
    # ---------------------------------------------------------------
    loader = TRECLoader()
    dataset = loader.load_split(split='train')

    np.random.seed(42)
    n_samples = min(1000, len(dataset['text']))
    indices = np.random.choice(len(dataset['text']), size=n_samples, replace=False)
    texts  = [dataset['text'][i] for i in indices]
    labels = [dataset['label'][i] for i in indices]
    y = np.array(labels, dtype=int)
    class_names = [f"Class_{i}" for i in range(50)]

    preprocessor = TRECPreprocessor(
        encoder_model=cfg['data']['encoder_model'],
        n_features_lda=cfg['data']['n_features_lda'],
    )
    print(f"Running Classical Feature Extraction into {cfg['data']['n_features_lda']} dims...")
    X = preprocessor.fit_transform(texts, y)  # (N, 32)

    # ---------------------------------------------------------------
    print("\n=== Phase 2: Quantum Honey Badger Feature Selection ===")
    # ---------------------------------------------------------------
    qpn_trainer = QPNMasterTrainer(
        n_features=X.shape[1],
        epochs_per_eval=2,
        n_qubits=n_qubits,
    )

    # Override QHBA config from yaml (matching few_rel's 30 agents / 100 iters)
    qpn_trainer.qhba.cfg = QHBAConfig(
        n_agents=qh_cfg['n_agents'],
        max_iter=qh_cfg['max_iter'],
        use_quantum_oracle=qh_cfg['use_quantum_oracle'],
    )
    qhba_result = qpn_trainer.fit(X, y)

    best_features = qhba_result.selected_indices
    print(f"Best Feature Mask found by QHBA: {best_features}")

    # ---------------------------------------------------------------
    # Fit QuantumFeaturePreprocessor (mirrors few_rel's qpn_preprocessor.fit())
    # Pads QHBA-selected features to 2^n_qubits and L2-normalizes.
    # ---------------------------------------------------------------
    qpn_preprocessor = QuantumFeaturePreprocessor(n_qubits=n_qubits)
    qpn_preprocessor.fit(X, np.array(best_features))
    print(
        f"QuantumFeaturePreprocessor fitted: "
        f"{len(best_features)} QHBA features -> {qpn_preprocessor.state_dim}-dim amplitude states"
    )

    # ---------------------------------------------------------------
    print("\n=== Phase 3: Episodic Meta-Training (QPN) ===")
    # ---------------------------------------------------------------
    model = QuantumProtoNet(
        n_qubits=n_qubits,
        init_type=init_type,
        cost_type=cost_type,
    )

    model = meta_train_qpn(
        model=model,
        X_train=X,
        y_train=y,
        global_preprocessor=qpn_preprocessor,
        n_way=t_cfg['n_way'],
        k_shot=t_cfg['k_shot'],
        n_query=t_cfg['n_query'],
        n_train_episodes=t_cfg['n_episodes'],
        lr=t_cfg['learning_rate'],
        lr_step_size=t_cfg['lr_step_size'],
        lr_gamma=t_cfg['lr_gamma'],
    )

    # ---------------------------------------------------------------
    print("\n=== Phase 4: Final Evaluation ===")
    # ---------------------------------------------------------------
    test_pool = build_class_pool(X, y)
    sampler   = EpisodeSampler(test_pool, e_cfg['n_way'], e_cfg['k_shot'], e_cfg['n_query'])

    def model_fn(ep):
        from data.episode_sampler import Episode
        s_x_enc = qpn_preprocessor.transform(ep.support_x)
        q_x_enc = qpn_preprocessor.transform(ep.query_x)
        encoded_ep = Episode(
            support_x=s_x_enc,
            support_y=ep.support_y,
            query_x=q_x_enc,
            query_y=ep.query_y,
        )
        return model.predict(encoded_ep)

    metrics = evaluate(model_fn, sampler, n_episodes=e_cfg['n_episodes'])

    print("\nTest Results:")
    print(f"  Accuracy:    {metrics['accuracy']['mean']:.2f}% ± {metrics['accuracy']['ci95']:.2f}%")
    print(f"  Weighted F1: {metrics['f1_weighted']['mean']:.2f}% ± {metrics['f1_weighted']['ci95']:.2f}%")

    # ---------------------------------------------------------------
    print("\n=== Phase 5: Generating Visualizations ===")
    # ---------------------------------------------------------------
    model.eval()
    ep     = sampler.sample(1)[0]
    y_true = ep.query_y

    s_x_enc = qpn_preprocessor.transform(ep.support_x)
    q_x_enc = qpn_preprocessor.transform(ep.query_x)

    from data.episode_sampler import Episode as Ep
    encoded_ep = Ep(support_x=s_x_enc, support_y=ep.support_y,
                    query_x=q_x_enc, query_y=ep.query_y)
    y_pred = model.predict(encoded_ep)

    evaluator = QuantumEvaluator(class_names=class_names)
    evaluator.plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png")

    import torch
    from quantum.training.qpn_model import FidelityParamShift
    s_x_t = torch.tensor(s_x_enc, dtype=torch.float32)
    q_x_t = torch.tensor(q_x_enc, dtype=torch.float32)
    theta_np = model.theta.detach().cpu().numpy()

    s_svs = FidelityParamShift._encode_batch(s_x_t, model._ansatz, theta_np, n_qubits)
    q_svs = FidelityParamShift._encode_batch(q_x_t, model._ansatz, theta_np, n_qubits)
    all_svs = s_svs + q_svs
    sv_data = [np.concatenate((sv.data.real, sv.data.imag)) for sv in all_svs]
    all_y   = np.concatenate((ep.support_y, ep.query_y))

    visualizer = QuantumSpaceVisualizer()
    visualizer.plot_tsne(sv_data, all_y, class_names, save_path="tsne_plot.png")

    print("Visualizations saved to confusion_matrix.png and tsne_plot.png")
    print("=== Pipeline Execution Complete ===")


if __name__ == "__main__":
    main()
