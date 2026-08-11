"""
test_trec_loader.py
Unit tests for the TRECLoader class (TREC-50 / FastFit/trec_50).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.trec_loader import TRECLoader


class TestTRECLoader:
    """Tests for TRECLoader — no network access required."""

    # ------------------------------------------------------------------
    # Label catalogue
    # ------------------------------------------------------------------

    def test_label_count_is_50(self):
        """TREC-50 must have exactly 50 fine-grained labels."""
        assert TRECLoader.n_classes() == 50
        assert len(TRECLoader.LABEL_NAMES) == 50

    def test_label_to_idx_is_inverse_of_label_names(self):
        """_LABEL_TO_IDX must be the exact inverse of LABEL_NAMES."""
        for idx, name in enumerate(TRECLoader.LABEL_NAMES):
            assert TRECLoader._LABEL_TO_IDX[name] == idx

    def test_no_duplicate_labels(self):
        """All 50 label strings must be unique."""
        assert len(set(TRECLoader.LABEL_NAMES)) == 50

    def test_label_format(self):
        """Every label must follow the 'COARSE:fine' format."""
        for name in TRECLoader.LABEL_NAMES:
            parts = name.split(":")
            assert len(parts) == 2, f"Bad format: {name}"
            assert parts[0].isupper(), f"Coarse not uppercase: {name}"
            assert len(parts[1]) > 0, f"Empty fine label: {name}"

    def test_coarse_groups_cover_all_labels(self):
        """All fine-grained labels must appear in exactly one coarse group."""
        all_in_groups = []
        for labels in TRECLoader.COARSE_GROUPS.values():
            all_in_groups.extend(labels)
        assert sorted(all_in_groups) == sorted(TRECLoader.LABEL_NAMES)

    def test_coarse_group_sizes(self):
        """Verify exact fine-grained counts per coarse group."""
        expected = {"ABBR": 2, "DESC": 4, "ENTY": 22, "HUM": 4, "LOC": 5, "NUM": 13}
        actual = {k: len(v) for k, v in TRECLoader.COARSE_GROUPS.items()}
        assert actual == expected

    # ------------------------------------------------------------------
    # Spot-check specific labels
    # ------------------------------------------------------------------

    def test_known_label_indices(self):
        """Spot-check a selection of known label → index mappings."""
        assert TRECLoader._LABEL_TO_IDX["ABBR:abb"] == 0
        assert TRECLoader._LABEL_TO_IDX["ABBR:exp"] == 1
        assert TRECLoader._LABEL_TO_IDX["ENTY:animal"] == 6
        assert TRECLoader._LABEL_TO_IDX["HUM:ind"] == 30
        assert TRECLoader._LABEL_TO_IDX["LOC:city"] == 32
        assert TRECLoader._LABEL_TO_IDX["NUM:date"] == 39
        assert TRECLoader._LABEL_TO_IDX["NUM:weight"] == 49

    def test_label_name_lookup(self):
        """label_name(idx) must be the inverse of _LABEL_TO_IDX."""
        assert TRECLoader.label_name(0) == "ABBR:abb"
        assert TRECLoader.label_name(6) == "ENTY:animal"
        assert TRECLoader.label_name(49) == "NUM:weight"

    def test_coarse_label_lookup(self):
        """coarse_label() must return only the coarse prefix."""
        assert TRECLoader.coarse_label(0) == "ABBR"
        assert TRECLoader.coarse_label(6) == "ENTY"
        assert TRECLoader.coarse_label(30) == "HUM"
        assert TRECLoader.coarse_label(49) == "NUM"

    # ------------------------------------------------------------------
    # Infra tests (no network)
    # ------------------------------------------------------------------

    def test_cache_dir_created(self, tmp_path):
        """Constructor must create the cache directory."""
        cache = tmp_path / "nested" / "cache"
        TRECLoader(cache_dir=str(cache))
        assert cache.exists()

    def test_repr(self, tmp_path):
        loader = TRECLoader(cache_dir=str(tmp_path))
        r = repr(loader)
        assert "TRECLoader" in r
        assert "FastFit/trec_50" in r

    def test_dataset_name(self):
        assert TRECLoader.DATASET_NAME == "FastFit/trec_50"

    # ------------------------------------------------------------------
    # Integration tests (require internet + HuggingFace download)
    # ------------------------------------------------------------------

    @pytest.mark.integration
    def test_load_returns_correct_structure(self, tmp_path):
        """Load both splits and verify dict structure."""
        loader = TRECLoader(cache_dir=str(tmp_path))
        train, test = loader.load()

        for split in (train, test):
            assert "text" in split
            assert "label" in split
            assert len(split["text"]) == len(split["label"])
            assert all(isinstance(t, str) for t in split["text"])
            assert all(0 <= lbl <= 49 for lbl in split["label"])

    @pytest.mark.integration
    def test_load_all_50_classes_present_in_train(self, tmp_path):
        """All 50 fine-grained classes should appear at least once in train."""
        loader = TRECLoader(cache_dir=str(tmp_path))
        train, _ = loader.load()
        unique_labels = set(train["label"])
        assert unique_labels == set(range(50)), (
            f"Missing classes: {set(range(50)) - unique_labels}"
        )

    @pytest.mark.integration
    def test_train_size(self, tmp_path):
        """TREC-50 train split should have 5452 samples (same questions as TREC-6)."""
        loader = TRECLoader(cache_dir=str(tmp_path))
        train, _ = loader.load()
        assert len(train["text"]) == 5452

    @pytest.mark.integration
    def test_test_size(self, tmp_path):
        """TREC-50 test split should have 500 samples."""
        loader = TRECLoader(cache_dir=str(tmp_path))
        _, test = loader.load()
        assert len(test["text"]) == 500
