from langgraph.graph import StateGraph, END

from .agents import (
    EcoShaadiState,
    classification_agent,
    logistics_agent,
    recommendation_agent,
    report_agent,
    sustainability_agent,
    vision_agent,
)


def build_graph():
    builder = StateGraph(EcoShaadiState)

    builder.add_node("vision", vision_agent)
    # NOTE: node ids below are deliberately NOT "classification" or "report" —
    # those exact strings are also keys in EcoShaadiState, and LangGraph
    # rejects a node whose name collides with a state field.
    builder.add_node("classify", classification_agent)
    builder.add_node("recommend", recommendation_agent)
    builder.add_node("logistics", logistics_agent)
    builder.add_node("sustainability", sustainability_agent)
    builder.add_node("generate_report", report_agent)

    builder.set_entry_point("vision")

    builder.add_edge("vision", "classify")
    builder.add_edge("classify", "recommend")
    builder.add_edge("recommend", "logistics")
    builder.add_edge("logistics", "sustainability")
    builder.add_edge("sustainability", "generate_report")
    builder.add_edge("generate_report", END)

    return builder.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
