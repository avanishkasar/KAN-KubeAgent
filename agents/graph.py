"""LangGraph wiring for the KAN-KubeAgent agent loop.

Flow (see LEARNING_GUIDE.md Level 2C and research/proposal/methodology_draft.md
Section 3): Metrics Watcher -> Loss-Curve Analyst / Cost Estimator ->
Supervisor (proposes) -> KAN Gate (decides, authoritative) -> Executor.
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from agents.executor import KANGatedExecutor
from agents.llm import propose_action
from agents.state import AgentState
from agents.trainjob_client import TrainJobClient
from kan_gate.features import extract_features
from kan_gate.gate import KANGate


def build_graph(client: TrainJobClient, gate: KANGate):
    executor = KANGatedExecutor(client)

    def metrics_watcher_node(state: AgentState) -> AgentState:
        status = client.get_status(state["trainjob_name"])
        return {
            **state,
            "loss_history": status.loss_history,
            "grad_norm_history": status.grad_norm_history,
            "lr": status.lr,
            "gpu_hours_used": status.gpu_hours_used,
            "gpu_hours_budget": status.gpu_hours_budget,
        }

    def loss_curve_and_cost_node(state: AgentState) -> AgentState:
        history = client.get_status(state["trainjob_name"]).to_history()
        features = extract_features(history)
        return {**state, "features": features}

    def supervisor_node(state: AgentState) -> AgentState:
        proposal = propose_action(state["features"], state["lr"])
        return {
            **state,
            "proposed_action": proposal["action"],
            "proposal_rationale": proposal["rationale"],
            "proposed_new_lr": proposal.get("new_lr"),
        }

    def kan_gate_node(state: AgentState) -> AgentState:
        result = gate.decide(state["features"])
        return {
            **state,
            "gate_decision": result.decision,
            "gate_score": result.score,
            "gate_formula": result.formula,
        }

    def execute_node(state: AgentState) -> AgentState:
        entry = executor.execute(
            state["trainjob_name"],
            state["gate_decision"],
            state["gate_score"],
            state["gate_formula"],
            new_lr=state.get("proposed_new_lr"),
        )
        entry["llm_proposal"] = state["proposed_action"]
        entry["llm_rationale"] = state["proposal_rationale"]
        entry["epoch_index"] = len(state["loss_history"]) - 1
        entry["features"] = state["features"]  # for kan_gate.real_run_logger's harvesting
        audit_log = state.get("audit_log", []) + [entry]
        return {**state, "executed": True, "audit_log": audit_log}

    def route_by_gate(state: AgentState) -> str:
        return "execute"  # every decision (including "continue") is logged

    workflow = StateGraph(AgentState)
    workflow.add_node("metrics_watcher", metrics_watcher_node)
    workflow.add_node("loss_curve_and_cost", loss_curve_and_cost_node)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("kan_gate", kan_gate_node)
    workflow.add_node("execute", execute_node)

    workflow.set_entry_point("metrics_watcher")
    workflow.add_edge("metrics_watcher", "loss_curve_and_cost")
    workflow.add_edge("loss_curve_and_cost", "supervisor")
    workflow.add_edge("supervisor", "kan_gate")
    workflow.add_conditional_edges("kan_gate", route_by_gate, {"execute": "execute"})
    workflow.add_edge("execute", END)

    return workflow.compile()
