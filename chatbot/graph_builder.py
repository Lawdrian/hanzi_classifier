# graph_builder.py
from typing import Any, Callable, Iterable, Optional
from langgraph.graph import StateGraph

class GraphBuilder:
    """
    Constructs StateGraph and renders it
    """

    def __init__(self, state_type: Any, checkpointer=None):
        self._graph = StateGraph(state_type)
        self._agent = None
        self._checkpointer = checkpointer

    def add_node(self, name: str, fn: Callable[..., dict]) -> None:
        self._graph.add_node(name, fn)

    def add_edge(self, src: str, dst: str) -> None:
        self._graph.add_edge(src, dst)

    def add_conditional_edges(self, src: str, cond: Callable[..., str], dsts: Iterable[str]) -> None:
        self._graph.add_conditional_edges(src, cond, list(dsts))

    # Build/compile
    def compile(self, interrupt_before=None) -> Any:
        """
        Compile the StateGraph and cache the compiled agent.
        Returns the compiled agent.
        """
        self._agent = self._graph.compile(checkpointer=self._checkpointer, interrupt_before=interrupt_before)
        return self._agent

    def get_agent(self) -> Optional[Any]:
        return self._agent

    # Lightweight rendering convenience
    def render_graph(self, agent: Optional[Any] = None, xray: bool = True) -> bytes:
        """
        Render the agent graph to PNG bytes. If `agent` is not provided,
        uses the compiled one cached via `compile()`.
        """
        target = agent or self._agent
        if target is None:
            raise ValueError("No agent provided and no compiled agent available. Call compile() first.")
        graph_obj = target.get_graph(xray=xray)
        # draw_mermaid_png typically returns bytes; raise if not available
        return graph_obj.draw_mermaid_png()