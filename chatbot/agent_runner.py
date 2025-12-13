# agent_runner.py
from dotenv import load_dotenv
load_dotenv()
from typing import Any, Dict
from graph_builder import GraphBuilder
from langgraph.graph import START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict, Annotated
from langchain.messages import AnyMessage, ToolMessage, AIMessage
import operator

# Import model factory and tools
from langchain.chat_models import init_chat_model
from tools import add, multiply, divide, request_classification

from classifier_mcp_client import ClassifierMCPClient
from agents import make_front_desk_agent, make_classify_node


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


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    # Hanzi classification state
    image: str
    classification_result: str
    classification_confidence: float

def make_tool_node(tools_by_name: Dict[str, Any]):
    """
    Return a node function `tool_node(state: dict) -> dict` that performs tool invocation
    for any tool call the LLM produced.
    """
    def tool_node(state: dict) -> dict:
        tool_call = state["messages"][-1].tool_calls[0]  # Single call
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        
        return {
            "messages": [ToolMessage(content=observation, tool_call_id=tool_call["id"])]
        }
    return tool_node


def router(state: MessagesState) -> str:
    """Route based on tool calls or transfer intent"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check if there are tool calls (getattr returns [] if no tool_calls attribute or empty list)
    tool_calls = getattr(last_message, "tool_calls", [])
    if tool_calls:
        tool_name = tool_calls[0]["name"]
        if tool_name == "request_classification":
            return "check_image_available"
        else:
            return "tool_node"
    return END


def check_image_available(state: MessagesState) -> dict:
    """Check if image is available for classification"""
    if state.get("image"):
        return {}
    else:
        return {
                "messages": [AIMessage(content="I'd like to classify that image for you. Please upload or take a photo of the Hanzi character.")]
            }


def should_classify_or_wait(state: MessagesState) -> str:
    """After checking image availability, decide next step"""
    print("should_classify_or_wait image:", state.get("image"))
    if state.get("image"):
        return "classify_node"
    else:
        # No image - interrupt and wait for user to upload
        return "wait_for_image"


def build_and_compile_agent(checkpointer=None):
    """Build graph with HITL nodes"""
    if checkpointer is None:
        checkpointer = InMemorySaver()

    model = _init_model()
    model_with_tools = model.bind_tools(MATH_TOOLS + HANZI_TOOLS)

    llm_node = make_front_desk_agent(model_with_tools)
    tool_node = make_tool_node(TOOLS_BY_NAME)
    
    # Initialize classifier client
    classifier_mcp_client = ClassifierMCPClient(base_url="http://localhost:8000")
    classify_node = make_classify_node(classifier_mcp_client)

    builder = GraphBuilder(MessagesState, checkpointer=checkpointer)
    
    # Add nodes
    builder.add_node("llm_call", llm_node)
    builder.add_node("tool_node", tool_node)
    builder.add_node("check_image_available", check_image_available)
    builder.add_node("classify_node", classify_node)
    builder.add_node("wait_for_image", lambda s: s)  # Dummy node for interrupt

    # Edges
    builder.add_edge(START, "llm_call")
    builder.add_conditional_edges(
        "llm_call",
        router,
        {
            "tool_node": "tool_node",
            "check_image_available": "check_image_available",
            END: END
        }
    )
    builder.add_edge("tool_node", "llm_call")

    # After checking image, decide: classify or wait
    builder.add_conditional_edges(
        "check_image_available",
        should_classify_or_wait,
        {
            "classify_node": "classify_node",
            "wait_for_image": "wait_for_image"
        }
    )

    builder.add_edge("classify_node", "llm_call")

    # Interrupt BEFORE wait_for_image so Streamlit can resume
    agent = builder.compile(interrupt_before=["wait_for_image"])
    return agent, builder