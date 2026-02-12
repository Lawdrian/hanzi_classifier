"""
MCP Server client for Hanzi classification.
Uses MCP over HTTP to call tools exposed by the classifier service.
"""

import asyncio
import json
from typing import Tuple, Dict, Any

from fastmcp import Client

DEFAULT_MCP_PATH = "/mcp"
DEFAULT_CONFIG = {
    "mcpServers": {
        "classifier": {
            "url": f"http://localhost:8001{DEFAULT_MCP_PATH}"
        },
        "translator": {
            "url": f"http://localhost:8000{DEFAULT_MCP_PATH}"
        }
    }
}

CLASSIFIER_TOOL = "classifier_classify_image"
TRANSLATOR_TOOL = "translator_translate_hanzi"



class MCPClient:
    """Client for MCP servers (classifier + translator)."""

    def __init__(self, config: Dict[str, Any] | None = None):
        """Initialize the MCP client with a multi-server config."""
        self._config = config or DEFAULT_CONFIG
        self.client = Client(self._config)

    def _run_sync(self, coro):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        if loop.is_running():
            raise RuntimeError("Event loop already running; use async methods instead.")

        return loop.run_until_complete(coro)

    @staticmethod
    def _extract_payload(result: Any) -> Any:
        if result is None:
            return None

        structured = getattr(result, "structured_content", None)
        if structured is not None:
            return structured

        content = getattr(result, "content", None)
        if content:
            first = content[0]
            text = getattr(first, "text", None)
            if text is not None:
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return text

        return result

    async def _call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        async with self.client:
            result = await self.client.call_tool(tool_name, arguments)
            return self._extract_payload(result)
    
    
    async def classify_async(self, image_base64: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify a Hanzi character image.
        
        Args:
            image_base64: Base64-encoded image string
            
        Returns:
            Tuple of (predicted_class, confidence, all_confidences_dict)
            
        Raises:
            Exception if classification fails
        """
        payload = {
            "image_base64": image_base64,
            "return_confidence": True,
        }

        data = await self._call_tool(CLASSIFIER_TOOL, {"request": payload})
        if not isinstance(data, dict):
            raise ValueError(f"Unexpected response format: {data}")

        try:
            return (
                data["predicted_class"],
                float(data["confidence"]),
                data["all_confidences"],
            )
        except KeyError as e:
            raise KeyError(f"Missing key in response: {e}")


    def classify(self, image_base64: str) -> Tuple[str, float, Dict[str, float]]:
        """Classify a Hanzi character image (sync wrapper)."""
        return self._run_sync(self.classify_async(image_base64))


    async def translate_async(self, text: str) -> str:
        """
        Translate Hanzi to English.

        Args:
            text: Hanzi text input

        Returns:
            Translated English text
        """
        data = await self._call_tool(TRANSLATOR_TOOL, {"text": text})
        if isinstance(data, str):
            return data
        if isinstance(data, dict) and "result" in data:
            return str(data["result"])
        raise ValueError(f"Unexpected response format: {data}")


    def translate(self, text: str) -> str:
        """Translate Hanzi to English (sync wrapper)."""
        return self._run_sync(self.translate_async(text))
    