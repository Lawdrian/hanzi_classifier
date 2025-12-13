"""
MCP Server client for Hanzi classification.
Handles HTTP communication with the classifier service.
"""

import requests
from typing import Tuple, Dict



class ClassifierMCPClient:
    """Client for the Hanzi Classifier MCP Server."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the classifier client.
        
        Args:
            base_url: URL of the MCP server (default: localhost:8000)
        """
        self.base_url = base_url
        self.classify_endpoint = f"{base_url}/classify"
        self.classes_endpoint = f"{base_url}/classes"
        self.health_endpoint = f"{base_url}/health"
    
    def health_check(self) -> bool:
        """Check if the classifier server is healthy."""
        try:
            response = requests.get(self.health_endpoint, timeout=5)
            response.raise_for_status()
            data = response.json()
            return data.get("classifier_loaded", False)
        except Exception as e:
            print(f"Health check failed: {e}")
            return False
    
    def classify(self, image_base64: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify a Hanzi character image.
        
        Args:
            image_base64: Base64-encoded image string
            
        Returns:
            Tuple of (predicted_class, confidence, all_confidences_dict)
            
        Raises:
            Exception if classification fails
        """
        try:
            payload = {
                "image_base64": image_base64,
                "return_confidence": True
            }
            
            print(f"Sending request to {self.classify_endpoint}")
            print(f"Payload keys: {payload.keys()}, image_base64 length: {len(image_base64)}")
            
            response = requests.post(self.classify_endpoint, json=payload, timeout=30)
            
            # Log response details before raising
            if not response.ok:
                print(f"Server error response: {response.status_code}")
                print(f"Response body: {response.text}")
            
            response.raise_for_status()
            
            data = response.json()
            return (
                data["predicted_class"],
                data["confidence"],
                data["all_confidences"]
            )
        except requests.exceptions.RequestException as e:
            print(f"Classification request failed: {e}")
            raise
        except KeyError as e:
            print(f"Unexpected response format: {e}")
            raise
    
    def get_classes(self) -> list[str]:
        """Get list of available character classes."""
        try:
            response = requests.get(self.classes_endpoint, timeout=5)
            response.raise_for_status()
            data = response.json()
            return data.get("classes", [])
        except Exception as e:
            print(f"Failed to get classes: {e}")
            raise
