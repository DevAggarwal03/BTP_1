"""
trec_loader.py
Download and load the TREC-50 fine-grained question-classification dataset
from HuggingFace (FastFit/trec_50).

TREC-50 maps questions to 50 fine-grained sub-categories across 6 coarse groups:

    ABBR (2):  abb, exp
    DESC (4):  def, desc, manner, reason
    ENTY (22): animal, body, color, cremat, currency, dismed, event, food,
               instru, lang, letter, other, plant, product, religion, sport,
               substance, symbol, techmth, termeq, veh, word
    HUM  (4):  desc, gr, ind, title
    LOC  (5):  city, country, mount, other, state
    NUM  (13): code, count, date, dist, money, ord, other, perc, period,
               speed, temp, volsize, weight

Label field in the raw dataset: string, e.g. ``"ENTY:animal"``.
This loader encodes them to contiguous integers 0–49 using a fixed
alphabetically-sorted mapping (see ``LABEL_NAMES``).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from datasets import load_dataset

if TYPE_CHECKING:
    from datasets import DatasetDict


class TRECLoader:
    """
    Loads the TREC-50 fine-grained question classification dataset
    from HuggingFace (``FastFit/trec_50``).

    Features
    --------
    - Downloads on first run; subsequent calls use the local disk cache.
    - Encodes string labels (e.g. ``"ENTY:animal"``) to integers 0–49.
    - Returns plain Python dicts for easy downstream use.

    Args:
        cache_dir: Directory to store the downloaded dataset.
                   Defaults to ``data/cache``.
    """

    DATASET_NAME: str = "FastFit/trec_50"

    # Fixed alphabetical ordering of all 50 fine-grained labels.
    # Index in this list == integer label returned by load().
    # Labels match the FastFit/trec_50 HuggingFace dataset format.
    LABEL_NAMES: list[str] = [
        "Abbreviation: Abbreviation.",                        #  0
        "Abbreviation: Expression abbreviated.",               #  1
        "Description: Definition of something.",               #  2
        "Description: Description of something.",              #  3
        "Description: Manner of an action.",                   #  4
        "Description: Reason.",                                #  5
        "Entity: Animal.",                                     #  6
        "Entity: Color.",                                      #  7
        "Entity: Currency name.",                              #  8
        "Entity: Disease and medicine.",                       #  9
        "Entity: Element and substance.",                      # 10
        "Entity: Equivalent term.",                            # 11
        "Entity: Event.",                                      # 12
        "Entity: Food.",                                       # 13
        "Entity: Invention, book and other creative piece.",   # 14
        "Entity: Language.",                                   # 15
        "Entity: Letter like a-z.",                            # 16
        "Entity: Musical instrument.",                         # 17
        "Entity: Organ of body.",                              # 18
        "Entity: Other entity.",                               # 19
        "Entity: Plant.",                                      # 20
        "Entity: Product.",                                    # 21
        "Entity: Religion.",                                   # 22
        "Entity: Sport.",                                      # 23
        "Entity: Symbols and sign.",                           # 24
        "Entity: Techniques and method.",                      # 25
        "Entity: Vehicle.",                                    # 26
        "Entity: Word with a special property.",               # 27
        "Human: Description of a person.",                     # 28
        "Human: Group or organization of persons.",            # 29
        "Human: Individual.",                                  # 30
        "Human: Title of a person.",                           # 31
        "Location: City.",                                     # 32
        "Location: Country.",                                  # 33
        "Location: Mountain.",                                 # 34
        "Location: Other location.",                           # 35
        "Location: State.",                                    # 36
        "Numeric: Date.",                                      # 37
        "Numeric: Distance, linear measure.",                  # 38
        "Numeric: Lasting time of something",                  # 39
        "Numeric: Number of something.",                       # 40
        "Numeric: Order, rank.",                               # 41
        "Numeric: Other number.",                              # 42
        "Numeric: Percent, fraction.",                         # 43
        "Numeric: Postcode or other code.",                    # 44
        "Numeric: Price.",                                     # 45
        "Numeric: Size, area and volume.",                     # 46
        "Numeric: Speed.",                                     # 47
        "Numeric: Temperature.",                               # 48
        "Numeric: Weight.",                                    # 49
    ]

    # Fast reverse-lookup: string label → int index
    _LABEL_TO_IDX: dict[str, int] = {
        name: idx for idx, name in enumerate(LABEL_NAMES)
    }

    # Coarse group prefix extracted from label string (text before ":")
    # Maps coarse name → list of fine-grained labels in that group.
    COARSE_GROUPS: dict[str, list[str]] = {
        "Abbreviation": [n for n in LABEL_NAMES if n.startswith("Abbreviation:")],
        "Description":  [n for n in LABEL_NAMES if n.startswith("Description:")],
        "Entity":       [n for n in LABEL_NAMES if n.startswith("Entity:")],
        "Human":        [n for n in LABEL_NAMES if n.startswith("Human:")],
        "Location":     [n for n in LABEL_NAMES if n.startswith("Location:")],
        "Numeric":      [n for n in LABEL_NAMES if n.startswith("Numeric:")],
    }

    def __init__(self, cache_dir: str = "data/cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> tuple[dict, dict]:
        """
        Load TREC-50 train and test splits.

        Returns
        -------
        train : dict
            Keys ``"text"`` (``List[str]``) and ``"label"`` (``List[int]``, range 0–49).
        test : dict
            Same structure as ``train``.
        """
        dataset: DatasetDict = load_dataset(
            self.DATASET_NAME,
            cache_dir=str(self.cache_dir),
        )

        train = self._to_dict(dataset["train"])
        test = self._to_dict(dataset["test"])

        return train, test

    def load_split(self, split: str = "train") -> dict:
        """
        Load a single split by name (``"train"`` or ``"test"``).

        Args:
            split: HuggingFace split identifier.

        Returns:
            dict with ``"text"`` and ``"label"`` lists.
        """
        dataset = load_dataset(
            self.DATASET_NAME,
            split=split,
            cache_dir=str(self.cache_dir),
        )
        return self._to_dict(dataset)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @classmethod
    def _to_dict(cls, split) -> dict:
        """
        Convert a HuggingFace Dataset split to a plain dict.

        The raw ``"label"`` field is a string (e.g. ``"ENTY:animal"``).
        We encode it to an integer using ``_LABEL_TO_IDX``.
        Unknown labels (should not occur in the standard dataset) raise KeyError.
        """
        texts: list[str] = []
        labels: list[int] = []

        for example in split:
            texts.append(example["text"])
            raw_label: str = example["label"]
            labels.append(cls._LABEL_TO_IDX[raw_label])

        return {"text": texts, "label": labels}

    @classmethod
    def label_name(cls, idx: int) -> str:
        """Return the fine-grained label string for an integer index (0–49)."""
        return cls.LABEL_NAMES[idx]

    @classmethod
    def coarse_label(cls, idx: int) -> str:
        """Return the coarse category name (e.g. ``"ENTY"``) for an index."""
        fine = cls.LABEL_NAMES[idx]
        return fine.split(":")[0]

    @classmethod
    def n_classes(cls) -> int:
        """Total number of fine-grained classes (50)."""
        return len(cls.LABEL_NAMES)

    def __repr__(self) -> str:
        return f"TRECLoader(dataset='{self.DATASET_NAME}', cache_dir='{self.cache_dir}')"
