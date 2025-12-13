from langchain.messages import SystemMessage, AIMessage
from typing import Callable
from classifier_mcp_client import ClassifierMCPClient


def make_front_desk_agent(model_with_tools) -> Callable[[dict], dict]:
    """
    Return a node function `front_desk_agent(state: dict) -> dict` bound to the provided model_with_tools.
    The node returns a dict with updated "messages" and increments "llm_calls".
    If transfer intent detected, also returns transaction details in state.
    """
    def front_desk_agent(state: dict) -> dict:
        print("llm_call called!")
        
        # Build system prompt dynamically based on image availability
        image_status = "An image has been uploaded and is available for classification." if state.get("image") else "No image has been uploaded yet."
        
        system = SystemMessage(content=f"""You are a helpful assistant. 
You can use tools one at a time. After using a tool, wait for the result.
You can:
- Help with arithmetic
- Chat with the user
- Perform a classification for hanzi character images

Image Status: {image_status}

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