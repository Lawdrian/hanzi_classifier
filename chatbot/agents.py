from langchain.messages import SystemMessage, AIMessage, HumanMessage, AnyMessage
from typing import Callable, Union, Literal, Any
from collections import deque
from mcp_client import MCPClient
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


class HanziTranslationParams(BaseModel):
    """Parameters for the Hanzi Translation subgraph"""
    text: str = Field(description="The Hanzi text to translate")


class CalculationParams(BaseModel):
    """Parameters for the Calculation subgraph"""
    operation: Literal["add", "multiply", "divide"] = Field(description="The mathematical operation to perform")
    operand1: float = Field(description="First operand")
    operand2: float = Field(description="Second operand")


class FinalResponseParams(BaseModel):
    """Parameters for final response to user (no subgraph needed)"""
    message: str = Field(description="The final response message to send to the user")


# ============================================================================
# DISPATCHER DECISION SCHEMA
# ============================================================================

class DispatchAction(BaseModel):
    """The dispatcher's decision on which subgraph to invoke"""
    reasoning: str = Field(description="Brief logic for choosing this subgraph")
    task: Union[AskUserParams, HanziClassificationParams, HanziTranslationParams, CalculationParams, FinalResponseParams] = Field(
        description="The parameters for the chosen subgraph or final response"
    )

    @property
    def target_node(self) -> str:
        """Route to the appropriate subgraph based on task type"""
        if isinstance(self.task, AskUserParams):
            return "ask_user_subgraph"
        if isinstance(self.task, HanziClassificationParams):
            return "hanzi_classification_subgraph"
        if isinstance(self.task, HanziTranslationParams):
            return "hanzi_translation_subgraph"
        if isinstance(self.task, CalculationParams):
            return "calculation_subgraph"
        if isinstance(self.task, FinalResponseParams):
            return "__end__"  # No subgraph needed, route to end
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
        
        system = SystemMessage(content=f"""You are a task orchestrator. Your job is to analyze user requests and create a clear, multi-step plan.

For each step, output a concise instruction describing:
1. What action to take 
2. Any relevant context
                               
Available actions:
- Ask User: Request clarification or additional information from the user.
- Hanzi Classification: Classify a Hanzi character from an image.
- Hanzi Translation: Translate given Hanzi characters to English.
- Calculation: Perform a mathematical operation (add, multiply, divide).
- Final Response: Send a final message back to the user.

Image available: {bool(state.get("image"))}
                               
Output the plan as a numbered list. Each step can only contain one action.
""")
        
        response = model.invoke([system] + state.get("messages", []))
        
        # Add metadata for UI filtering
        response.metadata = {
            "verbose_level": "user",  # "user", "info", or "debug"
            "agent": "orchestrator",
            "step": "planning"
        }
        
        # Extract the plan from the response
        plan_text = response.content if hasattr(response, 'content') else str(response)
        # Simple parsing: split by newlines and filter empty lines
        plan_steps = [step.strip() for step in plan_text.split('\n') if step.strip()]
        print("🎯 Orchestrator: Plan created:")
        print(plan_steps)
        return {
            "messages": [response],
            "plan_queue": deque(plan_steps),
            "llm_calls": state.get("llm_calls", 0) + 1
        }
    
    return orchestrator_agent


# ============================================================================
# EXECUTE STEP (DISPATCHER) AGENT
# ============================================================================

def make_execute_step(model) -> Callable[[dict, Any], dict]:
    """
    The Execute Step node dispatches each plan step to the appropriate subgraph.
    Returns a dict with 'next_node' field for conditional routing.
    """
    def execute_step(state: dict, config) -> dict:
        print("plan_queue:", state.get("plan_queue"))
        plan_queue = state.get("plan_queue", deque())
        
        # Check if we've finished all plan steps
        if not plan_queue:
            print("✅ Plan queue empty, routing to END")
            return {
                "next_node": "__end__",
                "plan_queue": plan_queue
            }
        
        current_step = plan_queue.popleft()
        print(f"📋 Execute Step: Processing '{current_step}'")
        messages = state.get("messages", [])
        image = state.get("image")
        current_output = state.get("current_output")
        
        # Build context for the dispatcher
        context = f"""Current step in the plan: {current_step}

Available context:
- User message: {messages[-1].content if messages and hasattr(messages[-1], 'content') else 'N/A'}
- Image available: {bool(image)}
    - Last step output: {current_output if current_output is not None else 'N/A'}

Based on this step, decide which subgraph to invoke and what parameters to use."""

        # Use structured output to get the dispatcher decision
        dispatcher = model.with_structured_output(DispatchAction)
        decision = dispatcher.invoke([
            {"role": "system", "content": "You are a semantic dispatcher. Analyze the plan step and choose the appropriate subgraph with validated parameters."},
            {"role": "user", "content": context}
        ])

        # Handle FinalResponseParams specially - add message before ending
        final_message = []
        if isinstance(decision.task, FinalResponseParams):
            msg = AIMessage(
                content=decision.task.message,
                metadata={
                    "verbose_level": "user",
                    "agent": "orchestrator",
                    "step": "final_response"
                }
            )
            final_message.append(msg)
            print(f"💬 Final Response: {decision.task.message[:100]}...")
        
        # Ensure we pass the real image bytes into the classification task
        active_task = decision.task.model_dump()
        if isinstance(decision.task, HanziClassificationParams) and image:
            active_task["image_base64"] = image

        # Return dict updating state with active_task, plan_queue, and next_node
        return {
            "active_task": active_task,
            "plan_queue": plan_queue,
            "llm_calls": state.get("llm_calls", 0) + 1,
            "next_node": decision.target_node,
            "messages": final_message
        }
    
    return execute_step



# ============================================================================
# SUBGRAPH: ASK USER (HITL)
# ============================================================================

class SubgraphInternalState(TypedDict):
    """Base state for subgraphs"""
    active_task: dict
    current_output: dict
    messages: list[AnyMessage]


class AskUserInternalState(SubgraphInternalState):
    """State for Ask User subgraph"""
    pass


def make_ask_user_subgraph_start(model) -> Callable[[AskUserInternalState], dict]:
    """
    Entry point for Ask User subgraph.
    Extracts the question and prompts the user.
    """
    def ask_user_start(state: AskUserInternalState) -> dict:
        task = state.get("active_task", {})
        question = task.get("question", "I need your input.")
        
        print(f"❓ Ask User: {question}")
        
        # Create message asking the user
        msg = AIMessage(
            content=question,
            metadata={
                "verbose_level": "user",
                "agent": "ask_user",
                "step": "hitl_question"
            }
        )
        
        return {
            "current_output": {
                "type": "ask_user",
                "status": "waiting",
                "question": question
            },
            "messages": [msg]
        }
    
    return ask_user_start


# ============================================================================
# SUBGRAPH: HANZI CLASSIFICATION
# ============================================================================

class HanziTranslationInternalState(SubgraphInternalState):
    """State for Hanzi Classification subgraph"""
    pass


def make_hanzi_classification_start() -> Callable[[HanziTranslationInternalState], dict]:
    """
    Entry point for Hanzi Classification subgraph.
    Validates the image and prepares for classification.
    """
    def hanzi_classification_start(state: HanziTranslationInternalState) -> dict:
        task = state.get("active_task", {})
        image = task.get("image_base64")
        
        if not image:
            return {
                "current_output": {
                    "type": "hanzi_classification",
                    "status": "error",
                    "error": "No image provided"
                }
            }
        
        print("🎯 Hanzi Classifier: Starting classification...")
        return {}
    
    return hanzi_classification_start


def make_classify_node(mcp_client: MCPClient) -> Callable[[HanziTranslationInternalState], dict]:
    """Perform image classification using the MCP server"""
    def classify_node(state: HanziTranslationInternalState) -> dict:
        task = state.get("active_task", {})
        image_base64 = task.get("image_base64")
        
        if not image_base64:
            return {
                "current_output": {                
                    "type": "hanzi_classification",
                    "status": "error",
                    "error": "No image provided"
                }
            }
        
        try:
            pred_class, confidence, all_confidences = mcp_client.classify(image_base64)
            
            langfuse = get_client()
            langfuse.score_current_trace(value=confidence, data_type="NUMERIC", name="classification_confidence")
            langfuse.score_current_trace(value=pred_class, data_type="CATEGORICAL", name="classification_result")
            langfuse.flush()
            
            print(f"✅ Hanzi Classifier: {pred_class} (confidence {confidence * 100:.1f}%)")
            
            # Create message with classification result
            msg = AIMessage(
                content=f"Classification: {pred_class} (confidence: {confidence*100:.1f}%)",
                metadata={
                    "verbose_level": "user",
                    "agent": "hanzi_classifier",
                    "step": "classification"
                }
            )
            
            return {
                "current_output": {
                    "type": "hanzi_classification",
                    "status": "ok",
                    "classification_result": pred_class,
                    "classification_confidence": float(confidence)
                },
                "messages": [msg]
            }
        except Exception as e:
            print(f"❌ Classification error: {e}")
            return {
                "current_output": {
                    "type": "hanzi_classification",
                    "status": "error",
                    "error": "Classification is not available"
                }
            }
    
    return classify_node


# ============================================================================
# SUBGRAPH: HANZI Translation
# ============================================================================

class HanziTranslationInternalState(SubgraphInternalState):
    """State for Hanzi Translation subgraph"""
    pass


def make_hanzi_translation_start() -> Callable[[HanziTranslationInternalState], dict]:
    """
    Entry point for Hanzi Translation subgraph.
    """
    def hanzi_translation_start(state: HanziTranslationInternalState) -> dict:
        task = state.get("active_task", {})
        text = task.get("text")
        
        if not text:
            return {
                "current_output": {
                    "type": "hanzi_translation",
                    "status": "error",
                    "error": "No text provided"
                }
            }
        
        print("🎯 Hanzi Translator: Starting translation...")
        return {}
    
    return hanzi_translation_start


def make_translation_node(mcp_client: MCPClient) -> Callable[[HanziTranslationInternalState], dict]:
    """Perform Hanzi translation using the MCP server"""
    def translation_node(state: HanziTranslationInternalState) -> dict:
        task = state.get("active_task", {})
        text = task.get("text")
        
        if not text:
            return {
                "current_output": {                
                    "type": "hanzi_translation",
                    "status": "error",
                    "error": "No text provided"
                }
            }
        
        try:
            result = mcp_client.translate(text)
            
            langfuse = get_client()
            langfuse.score_current_trace(value=result, name="translation_result")
            langfuse.flush()
            
            print(f"✅ Hanzi Translator: {result}")
            
            # Create message with translation result
            msg = AIMessage(
                content=f"Translation: {result}",
                metadata={
                    "verbose_level": "user",
                    "agent": "hanzi_translator",
                    "step": "translation"
                }
            )
            
            return {
                "current_output": {
                    "type": "hanzi_translation",
                    "status": "ok",
                    "translation_result": result
                },
                "messages": [msg]
            }
        except Exception as e:
            print(f"❌ Translation error: {e}")
            return {
                "current_output": {
                    "type": "hanzi_translation",
                    "status": "error",
                    "error": "Translation is not available"
                }
            }
    
    return translation_node


# ============================================================================
# SUBGRAPH: CALCULATION
# ============================================================================

class CalculationInternalState(SubgraphInternalState):
    """State for Calculation subgraph"""
    pass


def make_calculation_subgraph_start() -> Callable[[CalculationInternalState], dict]:
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


def make_calculation_executor() -> Callable[[CalculationInternalState], dict]:
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
                return {
                    "current_output": {
                        "type": "calculation",
                        "status": "error",
                        "error": "Division by zero"
                    }
                }
            result = operand1 / operand2
        
        print(f"🧮 Result: {result}")
        
        # Create message with calculation result
        msg = AIMessage(
            content=f"Calculation result: {result}",
            metadata={
                "verbose_level": "user",
                "agent": "calculator",
                "step": "calculation"
            }
        )
        
        return {
            "current_output": {
                "type": "calculation",
                "status": "ok",
                "result": result
            },
            "messages": [msg]
        }
    
    return calculate