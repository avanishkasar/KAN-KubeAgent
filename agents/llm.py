"""LLM-backed reasoning for the Supervisor agent (Anthropic API).

The LLM's job is strictly to PROPOSE an action and rationale from the
observed features - it never decides whether the action is safe/justified
to execute. That verification is the KAN gate's job alone (agents/graph.py
always calls the gate after this, and the gate's decision is authoritative
even when it disagrees with the LLM's proposal). See
research/proposal/methodology_draft.md Section 3 and LEARNING_GUIDE.md 3C.
"""
from __future__ import annotations

import json
import os

SYSTEM_PROMPT = """You are the Supervisor agent for KAN-KubeAgent, a system that \
watches a Kubeflow TrainJob's fine-tuning run and proposes control actions.

You will be given the current training features (loss plateau score, \
gradient trend, estimated benefit of decaying the learning rate, fraction \
of GPU-hour budget remaining, epochs since the last improvement).

Propose exactly one action: "continue", "adjust_lr", or "early_stop", with \
a one-sentence rationale. Your proposal is a suggestion only - final \
authority rests with a downstream KAN gate that scores the same features \
and can override you. Respond ONLY with JSON: \
{"action": "...", "rationale": "...", "new_lr": <float, only if action is adjust_lr>}"""


def _heuristic_fallback(features: dict[str, float]) -> dict:
    """Used when no ANTHROPIC_API_KEY is configured, so the agent loop is
    still runnable end-to-end for local development/testing."""
    from kan_gate.reference_policy import reference_decision

    action = reference_decision(features)
    return {
        "action": action,
        "rationale": "[heuristic fallback - no ANTHROPIC_API_KEY set] "
                      f"reference policy proposes '{action}' from plateau="
                      f"{features['loss_plateau_score']:.2f}",
        "new_lr": None,
    }


def propose_action(features: dict[str, float], current_lr: float, model: str = "claude-sonnet-5") -> dict:
    """Ask the LLM to propose a control action from the current features.

    Returns {"action": str, "rationale": str, "new_lr": float | None}.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return _heuristic_fallback(features)

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                f"Current features: {json.dumps(features, indent=2)}\n"
                f"Current learning rate: {current_lr}"
            ),
        }],
    )
    text = "".join(block.text for block in message.content if block.type == "text")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = _heuristic_fallback(features)
        parsed["rationale"] = f"[LLM response unparsable, fell back] raw: {text[:200]}"
    return parsed
