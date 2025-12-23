from langchain.messages import SystemMessage, AIMessage, HumanMessage
from typing import Callable, Union, Literal
from classifier_mcp_client import ClassifierMCPClient
from langfuse import get_client
from pydantic import BaseModel, Field
from langgraph.types import Command
from typing_extensions import TypedDict, Annotated
import operator


# ============================================================================
# PYDANTIC MODELS FOR SUBGRAPH INVOCATION PARAMETERS
# ============================================================================

class AskUserParams(BaseModel):
    """Parameters for the Ask User (HITL) subgraph"""
    question: str = Field(description="The question or clarification needed from the user")


class HanziClassificationParams(BaseModel):
    """Parameters for the Hanzi Classification subgraph"""
    image_base64: str = Field(description="Base64 encoded image for hanzi classification")


class CalculationParams(BaseModel):
    """Parameters for the Calculation subgraph"""
    operation: Literal["add", "multiply", "divide"] = Field(description="The mathematical operation to perform")
    operand1: float = Field(description="First operand")
    operand2: float = Field(description="Second operand")


# ============================================================================
# DISPATCHER DECISION SCHEMA
# ============================================================================

class DispatchAction(BaseModel):
    """The dispatcher's decision on which subgraph to invoke"""
    reasoning: str = Field(description="Brief logic for choosing this subgraph")
    task: Union[AskUserParams, HanziClassificationParams, CalculationParams] = Field(
        description="The parameters for the chosen subgraph"
    )

    @property
    def target_node(self) -> str:
        """Route to the appropriate subgraph based on task type"""
        if isinstance(self.task, AskUserParams):
            return "ask_user_subgraph"
        if isinstance(self.task, HanziClassificationParams):
            return "hanzi_classification_subgraph"
        if isinstance(self.task, CalculationParams):
            return "calculation_subgraph"
        return "error_node"


# ============================================================================
# ORCHESTRATOR (PLANNER) AGENT
# ============================================================================

def make_orchestrator_agent(model) -> Callable[[dict], dict]:
    """
    The Orchestrator creates a multi-step plan based on the user request.
    It analyzes the user's intent and breaks it down into actionable steps.
    """
    def orchestrator_agent(state: dict) -> dict:
        print("🎯 Orchestrator: Creating plan...")
        
        system = SystemMessage(content="""You are a task orchestrator. Your job is to analyze user requests and create a clear, multi-step plan.

For each step, output a concise instruction describing:
1. What action to take (e.g., "Ask user for clarification", "Classify the hanzi character", "Calculate sum of two numbers")
2. Any relevant context

Output the plan as a numbered list. Be specific about what needs to happen in each step.
""")
        
        response = model.invoke([system] + state.get("messages", []))
        
        # Extract the plan from the response
        plan_text = response.content if hasattr(response, 'content') else str(response)
        # Simple parsing: split by newlines and filter empty lines
        plan_steps = [step.strip() for step in plan_text.split('\n') if step.strip()]
        
        return {
            "messages": [response],
            "plan": plan_steps,
            "plan_index": 0,
            "llm_calls": state.get("llm_calls", 0) + 1
        }
    
    return orchestrator_agent


# ============================================================================
# EXECUTE STEP (DISPATCHER) AGENT
# ============================================================================

def make_execute_step(model) -> Callable[[dict], dict]:
    """
    The Execute Step node dispatches each plan step to the appropriate subgraph.
    It uses structured output to determine which subgraph to invoke and what parameters to pass.
    """
    def execute_step(state: dict, config) -> Command:
        print(f"📋 Execute Step: Processing plan step {state.get('plan_index', 0)}")
        
        plan = state.get("plan", [])
        plan_index = state.get("plan_index", 0)
        
        # Check if we've finished all plan steps
        if plan_index >= len(plan):
            return Command(goto="END")
        
        current_step = plan[plan_index]
        messages = state.get("messages", [])
        image = state.get("image")
        
        # Build context for the dispatcher
        context = f"""Current step in the plan: {current_step}

Available context:
- User message: {messages[-1].content if messages and hasattr(messages[-1], 'content') else 'N/A'}
- Image available: {bool(image)}

Based on this step, decide which subgraph to invoke and what parameters to use."""
        
        # Use structured output to get the dispatcher decision
        dispatcher = model.with_structured_output(DispatchAction)
        decision = dispatcher.invoke([
            {"role": "system", "content": "You are a semantic dispatcher. Analyze the plan step and choose the appropriate subgraph with validated parameters."},
            {"role": "user", "content": context}
        ])
        
        # Route to the chosen subgraph with parameters
        return Command(
            goto=decision.target_node,
            update={
                "active_task": decision.task.model_dump(),
                "plan_index": plan_index + 1,
                "llm_calls": state.get("llm_calls", 0) + 1
            }
        )
    
    return execute_step


# ============================================================================
# SUBGRAPH: ASK USER (HITL)
# ============================================================================

class AskUserInternalState(TypedDict):
    """State for Ask User subgraph"""
    active_task: dict  # Contains AskUserParams
    user_response: str


def make_ask_user_subgraph_start(model) -> Callable[[dict], dict]:
    """
    Entry point for Ask User subgraph.
    Extracts the question and prompts the user.
    """
    def ask_user_start(state: AskUserInternalState) -> dict:
        task = state.get("active_task", {})
        question = task.get("question", "I need your input.")
        
        print(f"❓ Ask User: {question}")
        
        return {
            "user_response": ""  # Will be filled by HITL interrupt
        }
    
    return ask_user_start


# ============================================================================
# SUBGRAPH: HANZI CLASSIFICATION
# ============================================================================

class HanziClassificationInternalState(TypedDict):
    """State for Hanzi Classification subgraph"""
    active_task: dict  # Contains HanziClassificationParams
    classification_result: str
    classification_confidence: float


def make_hanzi_classification_start() -> Callable[[dict], dict]:
    """
    Entry point for Hanzi Classification subgraph.
    Validates the image and prepares for classification.
    """
    def hanzi_classification_start(state: HanziClassificationInternalState) -> dict:
        task = state.get("active_task", {})
        image = task.get("image_base64")
        
        if not image:
            return {
                "classification_result": "ERROR",
                "classification_confidence": 0.0
            }
        
        print("🎯 Hanzi Classifier: Starting classification...")
        return {}
    
    return hanzi_classification_start


def make_classify_node_v2(classifier_mcp_client: ClassifierMCPClient) -> Callable[[dict], dict]:
    """Perform image classification using the MCP server (v2 - for new architecture)"""
    def classify_node(state: HanziClassificationInternalState) -> dict:
        task = state.get("active_task", {})
        image_base64 = task.get("image_base64")
        
        if not image_base64:
            return {
                "classification_result": "ERROR: No image",
                "classification_confidence": 0.0
            }
        
        try:
            pred_class, confidence, all_confidences = classifier_mcp_client.classify(image_base64)
            
            langfuse = get_client()
            langfuse.score_current_trace(value=confidence, data_type="NUMERIC", name="classification_confidence")
            langfuse.score_current_trace(value=pred_class, data_type="CATEGORICAL", name="classification_result")
            langfuse.flush()
            
            print(f"✅ Hanzi Classifier: {pred_class} (confidence {confidence * 100:.1f}%)")
            
            return {
                "classification_result": pred_class,
                "classification_confidence": float(confidence)
            }
        except Exception as e:
            print(f"❌ Classification error: {e}")
            return {
                "classification_result": f"ERROR: {str(e)}",
                "classification_confidence": 0.0
            }
    
    return classify_node


# ============================================================================
# SUBGRAPH: CALCULATION
# ============================================================================

class CalculationInternalState(TypedDict):
    """State for Calculation subgraph"""
    active_task: dict  # Contains CalculationParams
    calculation_result: float


def make_calculation_subgraph_start() -> Callable[[dict], dict]:
    """
    Entry point for Calculation subgraph.
    Extracts and validates the operation parameters.
    """
    def calculation_start(state: CalculationInternalState) -> dict:
        task = state.get("active_task", {})
        operation = task.get("operation")
        operand1 = task.get("operand1")
        operand2 = task.get("operand2")
        
        print(f"🧮 Calculator: {operand1} {operation} {operand2}")
        
        return {}
    
    return calculation_start


def make_calculation_executor() -> Callable[[dict], dict]:
    """
    Execute the actual calculation based on the operation.
    """
    def calculate(state: CalculationInternalState) -> dict:
        task = state.get("active_task", {})
        operation = task.get("operation")
        operand1 = task.get("operand1", 0)
        operand2 = task.get("operand2", 0)
        
        result = 0
        if operation == "add":
            result = operand1 + operand2
        elif operation == "multiply":
            result = operand1 * operand2
        elif operation == "divide":
            if operand2 == 0:
                return {"calculation_result": float('inf')}
            result = operand1 / operand2
        
        print(f"🧮 Result: {result}")
        
        return {"calculation_result": result}
    
    return calculate


# ============================================================================
# LEGACY FUNCTIONS (DEPRECATED - kept for backward compatibility)
# ============================================================================

def make_front_desk_agent(model_with_tools) -> Callable[[dict], dict]:
    """
    [DEPRECATED] Legacy front desk agent - replaced by orchestrator + dispatcher pattern.
    Return a node function `front_desk_agent(state: dict) -> dict` bound to the provided model_with_tools.
    """
    def front_desk_agent(state: dict) -> dict:
        print("llm_call called!")
        
        system = SystemMessage(content=f"""You are a helpful assistant. 
You can use tools one at a time. After using a tool, wait for the result.
You can:
- Help with arithmetic
- Chat with the user
- Perform a classification for hanzi character images

When a user wants to classify a hanzi image:
1. Call the appropriate tool to invoke the classification

IMPORTANT: When you see a classification result, do not invoke the classification tool again, but instead tell the user the result you got.
""")
        response = model_with_tools.invoke([system] + state["messages"])
        
        result = {
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1
        }
        
        return result
    return front_desk_agent


def make_classify_node(classifier_mcp_client: ClassifierMCPClient) -> Callable[[dict], dict]:
    """
    [DEPRECATED] Legacy classification node - replaced by make_classify_node_v2().
    """
    def classify_node(state: dict) -> dict:
        """Perform image classification using the MCP server."""
        
        image_base64 = state.get("image")
        
        if not image_base64:
            return {
                "messages": [AIMessage(content="Error: No image available for classification.")],
                "classification_result": None,
                "classification_confidence": 0.0
            }
        
        try:
            # Call the classifier server
            pred_class, confidence, all_confidences = classifier_mcp_client.classify(image_base64)
            
            # Format confidence scores for display
            confidence_str = f"{confidence * 100:.1f}%"
            
            langfuse = get_client()
            langfuse.score_current_trace(value=confidence, data_type="NUMERIC", name="classification_confidence")
            langfuse.score_current_trace(value=pred_class, data_type="CATEGORICAL", name="classification_result")
            langfuse.flush()
            print("Logged classification results to Langfuse.")
            return {
                "messages": [AIMessage(content=f"Result Classified: '{pred_class}' (confidence {confidence_str})")],
                "classification_result": pred_class,
                "classification_confidence": float(confidence)
            }
        except Exception as e:
            print(f"Classification error: {e}")
            return {
                "messages": [AIMessage(content=f"Error during classification: {str(e)}")],
                "classification_result": None,
                "classification_confidence": 0.0
            }
    return classify_node