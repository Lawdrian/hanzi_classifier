import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test():
    server_params = StdioServerParameters(
        command="uv", 
        args=["run", "python", "server.py"]
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Discover all available tools
            tools = await session.list_tools()
            print("Available tools:")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description}")
                print(f"    Schema: {tool.inputSchema}")
            
            # Call the tool
            result = await session.call_tool("translate_hanzi", {"text": "你好。我叫瑞恩。你有什麽興趣？"})
            print(f"\nTranslation result: {result}")

asyncio.run(test())