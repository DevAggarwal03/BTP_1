"""Small, explicit configuration objects for the experiment."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EpisodeConfig:
    """Definition of one few-shot episode.

    With the default 5-way/1-shot/15-query setting, an episode contains
    5 support examples and 75 query examples (80 examples total).
    """

    n_way: int = 5
    k_shot: int = 1
    q_queries: int = 15

    @property
    def support_size(self) -> int:
        return self.n_way * self.k_shot

    @property
    def query_size(self) -> int:
        return self.n_way * self.q_queries

    @property
    def total_size(self) -> int:
        return self.support_size + self.query_size


@dataclass(frozen=True)
class ModelConfig:
    """Architecture settings."""

    embedding_dim: int = 384
    compression_hidden_dim: int = 64
    n_qubits: int = 4
    quantum_reps: int = 2
    compression_mode: str = "mlp"
    beta_init: float = 10.0


@dataclass(frozen=True)
class TrainConfig:
    """Training and reproducibility settings."""

    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    train_episodes: int = 20
    validation_episodes: int = 20
    seed: int = 7

