from fastmcp import FastMCP
from llama_cpp import Llama

# Initialize
mcp = FastMCP("Translator")
llm = Llama(
    model_path="./qwen3-0.6b.Q4_K_M_translator.gguf",
    n_ctx=1024,
    verbose=False
)

@mcp.tool()
def translate_hanzi(text: str) -> str:
    """Translates Hanzi to English using finetuned LLM"""

    prompt_template = f"""
### Instruction:
Translate to English

### Input:
{text}

### Response:
"""
    output = llm(prompt_template, max_tokens=1024, stop=["###"])
    content = output["choices"][0]["text"]
    return content

if __name__ == "__main__":
    mcp.run()