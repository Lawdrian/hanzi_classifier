# agent_runner.py
from dotenv import load_dotenv
load_dotenv()
from typing import Any, Dict
from collections import deque
from graph_builder import GraphBuilder
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict, Annotated
from langchain.messages import AnyMessage, ToolMessage, AIMessage
import operator

# Import model factory and tools
from langchain.chat_models import init_chat_model
from tools import add, multiply, divide, request_classification

from mcp_client import MCPClient
from agents import (
    make_hanzi_translation_start,
    make_orchestrator_agent,
    make_execute_step,
    make_ask_user_subgraph_start,
    make_hanzi_classification_start,
    make_classify_node,
    make_calculation_subgraph_start,
    make_calculation_executor,
    AskUserInternalState,
    HanziTranslationInternalState,
    CalculationInternalState,
    make_translation_node,
)


def _init_model():
    #return init_chat_model(
    #    model="gemini-2.5-flash-lite",
    #    model_provider="google_genai",
    #    temperature=0
    #)
    return init_chat_model(
        model="openai/gpt-oss-120b",
        model_provider="groq",
        temperature=0
    )

# Prepare tools list and mapping (tools are the decorated tool objects)
MATH_TOOLS = [add, multiply, divide]
HANZI_TOOLS = [request_classification]  # Only read-only tools for the LLM
TOOLS_BY_NAME = {t.name: t for t in MATH_TOOLS + HANZI_TOOLS}


# ============================================================================
# MAIN GRAPH STATE
# ============================================================================

class MainState(TypedDict):
    """Main orchestrator state"""
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    plan_queue: deque  # Queue of plan steps
    active_task: dict  # Parameters for the current subgraph
    current_output: dict # Output from current plan step execution
    image: str  # Base64 encoded image for classification
    next_node: str  # Target node for conditional routing


# ============================================================================
# SUBGRAPH BUILDERS
# ============================================================================

def build_ask_user_subgraph(model) -> StateGraph:
    """Build the Ask User (HITL) subgraph"""
    graph = StateGraph(AskUserInternalState)
    
    ask_user_start = make_ask_user_subgraph_start(model)
    graph.add_node("ask_user_start", ask_user_start)  # type: ignore[arg-type]

    graph.add_edge(START, "ask_user_start")
    graph.add_edge("ask_user_start", END)
    
    return graph.compile() # type: ignore[arg-type]


def build_hanzi_classification_subgraph(classifier_mcp_client: MCPClient) -> StateGraph:
    """Build the Hanzi Classification subgraph"""
    graph = StateGraph(HanziTranslationInternalState)
    
    hanzi_start = make_hanzi_classification_start()
    classify_node = make_classify_node(classifier_mcp_client)
    
    graph.add_node("hanzi_start", hanzi_start)  # type: ignore[arg-type]
    graph.add_node("classify", classify_node)  # type: ignore[arg-type]
    
    graph.add_edge(START, "hanzi_start")
    graph.add_edge("hanzi_start", "classify")
    graph.add_edge("classify", END)
    
    return graph.compile() # type: ignore[arg-type]


def build_hanzi_translation_subgraph(classifier_mcp_client: MCPClient) -> StateGraph:
    """Build the Hanzi Translation subgraph"""
    graph = StateGraph(HanziTranslationInternalState)
    
    hanzi_start = make_hanzi_translation_start()
    translation_node = make_translation_node(classifier_mcp_client)
    
    graph.add_node("hanzi_start", hanzi_start)  # type: ignore[arg-type]
    graph.add_node("translation", translation_node)  # type: ignore[arg-type]
    
    graph.add_edge(START, "hanzi_start")
    graph.add_edge("hanzi_start", "translation")
    graph.add_edge("translation", END)
    
    return graph.compile() # type: ignore[arg-type]


def build_calculation_subgraph() -> StateGraph:
    """Build the Calculation subgraph"""
    graph = StateGraph(CalculationInternalState)
    
    calc_start = make_calculation_subgraph_start()
    calc_executor = make_calculation_executor()
    
    graph.add_node("calc_start", calc_start)  # type: ignore[arg-type]
    graph.add_node("execute_calc", calc_executor)  # type: ignore[arg-type]
    
    graph.add_edge(START, "calc_start")
    graph.add_edge("calc_start", "execute_calc")
    graph.add_edge("execute_calc", END)
    
    return graph.compile() # type: ignore[arg-type]


# ============================================================================
# GRAPH COMPOSITION
# ============================================================================

def build_and_compile_agent(checkpointer=None):
    """
    Build the new orchestrator-dispatcher architecture with three subgraphs:
    1. Orchestrator: Creates multi-step plan from user request
    2. Execute Step: Routes each plan step to appropriate subgraph
    3. Subgraphs: Ask User, Hanzi Classification, Calculation
    """
    if checkpointer is None:
        checkpointer = InMemorySaver()

    model = _init_model()

    # Initialize classifier client
    classifier_mcp_client = MCPClient()

    # Build subgraphs
    ask_user_subgraph = build_ask_user_subgraph(model)
    hanzi_classification_subgraph = build_hanzi_classification_subgraph(classifier_mcp_client)
    hanzi_translation_subgraph = build_hanzi_translation_subgraph(classifier_mcp_client)
    calculation_subgraph = build_calculation_subgraph()
    

    # Create main graph builder
    builder = GraphBuilder(MainState, checkpointer=checkpointer)

    # Add main nodes
    orchestrator = make_orchestrator_agent(model)
    execute_step = make_execute_step(model)
    
    builder.add_node("orchestrator", orchestrator)
    builder.add_node("execute_step", execute_step) # type: ignore[arg-type]
    
    # Add subgraph nodes
    builder.add_node("ask_user_subgraph", ask_user_subgraph) # type: ignore[arg-type]
    builder.add_node("hanzi_classification_subgraph", hanzi_classification_subgraph) # type: ignore[arg-type]
    builder.add_node("calculation_subgraph", calculation_subgraph) # type: ignore[arg-type]
    builder.add_node("wait_for_user", lambda s: s) # type: ignore[arg-type]
    builder.add_node("hanzi_translation_subgraph", hanzi_translation_subgraph) # type: ignore[arg-type]

    # Main flow: START -> Orchestrator -> Execute Step -> Subgraphs
    builder.add_edge(START, "orchestrator")
    builder.add_edge("orchestrator", "execute_step")
    
    # Conditional routing from execute_step to subgraphs based on dispatcher decision
    builder.add_conditional_edges(
        "execute_step",
        lambda state: state["next_node"],  # Read target node from state
        {
            "ask_user_subgraph": "ask_user_subgraph",
            "hanzi_classification_subgraph": "hanzi_classification_subgraph",
            "hanzi_translation_subgraph": "hanzi_translation_subgraph",
            "calculation_subgraph": "calculation_subgraph",
            "__end__": END,
        }
    )
    
    # Subgraphs route back to execute_step to process next plan step
    builder.add_edge("ask_user_subgraph", "wait_for_user")
    builder.add_edge("wait_for_user", "execute_step")
    builder.add_edge("hanzi_classification_subgraph", "execute_step")
    builder.add_edge("calculation_subgraph", "execute_step")
    builder.add_edge("hanzi_translation_subgraph", "execute_step")
    agent = builder.compile(interrupt_before=["wait_for_user"])
    return agent, builder
