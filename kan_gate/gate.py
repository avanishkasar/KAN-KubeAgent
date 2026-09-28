"""The KAN gate: the interpretable scoring/decision layer every proposed
training-control action must pass through before it executes.

See research/proposal/methodology_draft.md Section 4.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from kan_gate.features import FEATURE_NAMES

DEFAULT_CKPT_PATH = Path(__file__).parent / "checkpoints" / "kan_gate"

# Decision bands on the 0-100 stop score.
CONTINUE_MAX = 40
ADJUST_LR_MAX = 75


@dataclass
class GateResult:
    decision: str          # "continue" | "adjust_lr" | "early_stop"
    score: float            # 0-100 stop score
    formula: str             # human-readable symbolic formula
    features: dict[str, float]


def _score_to_decision(score: float) -> str:
    if score < CONTINUE_MAX:
        return "continue"
    if score < ADJUST_LR_MAX:
        return "adjust_lr"
    return "early_stop"


class KANGate:
    """Wraps a pykan KAN model as the project's interpretable gate.

    Every proposed action from the agent layer is scored here before it is
    ever applied to a real TrainJob (kan_gate.gate.KANGate.decide ->
    agents.executor.KANGatedExecutor.execute).
    """

    def __init__(self, width: list[int] | None = None, grid: int = 5, k: int = 3, seed: int = 42):
        # Kept intentionally narrow ([5, 3, 1], not [5, 4, 3, 1]): a wider/
        # deeper KAN composes many more sin/exp terms during symbolic
        # extraction, which is accurate but unreadable. A narrower gate
        # trades a little capacity for the short, auditable formula that
        # is this project's actual point - see methodology_draft.md 4.3.
        self.width = width or [len(FEATURE_NAMES), 3, 1]
        self.model = self._build_model(grid=grid, k=k, seed=seed)
        self._formula_cache: str | None = None

    def _build_model(self, grid: int, k: int, seed: int):
        from kan import KAN

        ckpt_dir = DEFAULT_CKPT_PATH.parent / "_tmp"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return KAN(width=self.width, grid=grid, k=k, seed=seed, auto_save=False,
                   ckpt_path=str(ckpt_dir))

    def encode(self, features: dict[str, float]) -> torch.Tensor:
        missing = [name for name in FEATURE_NAMES if name not in features]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        return torch.tensor([[features[name] for name in FEATURE_NAMES]], dtype=torch.float32)

    def fit(self, X: torch.Tensor, y: torch.Tensor, steps: int = 200, lamb: float = 0.001,
            test_X: torch.Tensor | None = None, test_y: torch.Tensor | None = None,
            prune: bool = True) -> None:
        """Train the gate on labeled (features -> stop score) pairs.

        `lamb` (L1) and lamb_entropy push the model toward a sparse set of
        active edges, and `prune` then drops the near-zero ones - both
        needed to get a short, readable formula instead of an unpruned
        KAN's dense symbolic expression. See methodology_draft.md 4.2-4.3.
        """
        dataset = {
            "train_input": X,
            "train_label": y,
            "test_input": test_X if test_X is not None else X,
            "test_label": test_y if test_y is not None else y,
        }
        self.model.fit(dataset, opt="Adam", steps=steps, lamb=lamb, lamb_entropy=2.0)
        if prune:
            pruned = self.model.prune()
            if pruned is not None:
                self.model = pruned
            # A short re-fit stabilises the pruned model's remaining edges.
            # update_grid=False: re-gridding a heavily pruned model can hit
            # a degenerate least-squares fit (too few active edges/points).
            self.model.fit(dataset, opt="Adam", steps=max(20, steps // 5), lamb=lamb,
                            update_grid=False)
        self._formula_cache = None

    def raw_score(self, features: dict[str, float]) -> float:
        x = self.encode(features)
        with torch.no_grad():
            raw = self.model(x)
        # The model is trained directly against 0-100 score labels (see
        # reference_policy.decision_to_score), so its raw output already IS
        # the score - no extra squashing here. An earlier version applied
        # sigmoid(raw)*100 on top of that, which silently saturated nearly
        # every decision to ~100 once the raw output exceeded ~5. Clamp
        # only to guard against out-of-range extrapolation.
        return max(0.0, min(100.0, raw.item()))

    def decide(self, features: dict[str, float]) -> GateResult:
        score = self.raw_score(features)
        return GateResult(
            decision=_score_to_decision(score),
            score=score,
            formula=self.formula(),
            features=features,
        )

    def formula(self, lib: list[str] | None = None) -> str:
        """Extract the current symbolic formula. Cached after the first call
        (or after fit()) since auto_symbolic is relatively expensive."""
        if self._formula_cache is not None:
            return self._formula_cache
        # A restrained library (no exp/log) keeps the fitted symbolic terms
        # from compounding into unreadable nested expressions.
        lib = lib or ["x", "x^2", "x^3", "sin"]
        try:
            self.model.auto_symbolic(lib=lib, verbose=0)
            formulas = self.model.symbolic_formula(var=FEATURE_NAMES)
            expr = str(formulas[0][0]) if formulas and formulas[0] else "stop_score = <unavailable>"
        except Exception as exc:  # symbolic regression can fail on untrained/odd models
            expr = f"stop_score = <symbolic extraction failed: {exc}>"
        self._formula_cache = f"stop_score = {expr}"
        return self._formula_cache

    def save(self, path: Path | str = DEFAULT_CKPT_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.saveckpt(str(path))

    @classmethod
    def load(cls, path: Path | str = DEFAULT_CKPT_PATH, width: list[int] | None = None) -> "KANGate":
        from kan import KAN

        gate = cls.__new__(cls)
        gate.width = width or [len(FEATURE_NAMES), 4, 3, 1]
        gate.model = KAN.loadckpt(str(path))
        gate._formula_cache = None
        return gate
