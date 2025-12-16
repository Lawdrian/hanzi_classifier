"""
main.py — Streamlit UI and wiring
graph_builder.py — constructs and compiles the langgraph StateGraph
tools.py — tool definitions (pure functions, small wrappers)
agent_runner.py — functions to invoke compiled agent and return outputs
state_store.py — tiny wrapper around st.session_state
visualize.py — generates PNG bytes for st.image
config.py — env var parsing
requirements.txt, README.md, tests under tests/
"""

import uuid
import streamlit as st
from langchain.messages import HumanMessage, AIMessage, ToolMessage
from agent_runner import build_and_compile_agent
import base64
from langfuse import get_client, propagate_attributes, Langfuse
from langfuse.langchain import CallbackHandler

st.set_page_config(page_title="LangGraph Hanzi Agent", layout="centered")
LANGFUSE_HANDLER = CallbackHandler()

@st.cache_resource
def get_agent_and_builder():
    return build_and_compile_agent()


def render_chat_history(messages: list):
    """
    Directly renders the LangGraph message history using Streamlit chat elements.
    """
    
    # Iterate through the LangGraph message history
    for msg in messages:
        
        #  1. User Messages
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)

        # 2. AI Messages (LLM messages + TOOL Invokations)
        elif isinstance(msg, AIMessage):
            # If the AI has tool calls, render them nicely
            if msg.tool_calls:
                with st.chat_message("assistant"):
                    # Create an expander for "Thought Process" to keep UI clean
                    with st.status("🤖 Agent is thinking...", expanded=False) as status:
                        for tool in msg.tool_calls:
                            st.write(f"**Calling Tool:** `{tool['name']}`")
                            st.json(tool['args'])
                        status.update(label="✅ Tools invoked", state="complete")
                    
                    # If there is also text content along with the tool call
                    if msg.content:
                        st.markdown(msg.content)
            
            # Normal AI response without tools
            elif msg.content:
                with st.chat_message("assistant"):
                    st.markdown(msg.content)

        # 3. Tool Outputs
        elif isinstance(msg, ToolMessage): # TODO Decide if actually wanna show or not
            with st.expander(f"Tool Output: {msg.name}"):
                 st.code(msg.content)


def run_agent_sync(agent, langfuse: Langfuse, message: str|None, config, thread_id: str):
    """
    Run the agent synchronously.
    """
    with langfuse.start_as_current_observation(as_type="span", name="langgraph_invoke") as span:
        # Propagate session_id to all observations
        with propagate_attributes(session_id=thread_id):
            # Pass handler to the chain invocation
            content = {"messages": [HumanMessage(message)]} if message else None
            span.update(input=content)        
            result = agent.invoke(content, config)
            span.update(output=result)
            
            return result


def sidebar(agent=None, langfuse: Langfuse = None, thread_id: str | None = None):
    st.sidebar.header("Upload Image")
    
    upload_method = st.sidebar.radio("Choose input method:", ["Upload File", "Take Photo"])
    
    image_bytes = None
    if upload_method == "Upload File":
        uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image_bytes = uploaded_file.read()
            mime_type = "image/jpeg"
    else:
        camera_photo = st.sidebar.camera_input("Take a photo")
        if camera_photo:
            image_bytes = camera_photo.getvalue()
            mime_type = "image/jpeg"
    
    # If image uploaded, convert to base64 and store
    if image_bytes:
        image_base64 = base64.b64encode(image_bytes).decode()
        image_data_uri = f"data:{mime_type};base64,{image_base64}"
        st.sidebar.success("✅ Image uploaded!")
        st.sidebar.image(image_bytes, caption="Current image")
        
        # If graph was waiting for image, push to agent state and resume
        if agent and thread_id:
            try:
                base_config = {"configurable": {"thread_id": thread_id}}
                cb_config = base_config.copy()
                cb_config["callbacks"] = [LANGFUSE_HANDLER]
                current_state = agent.get_state(base_config)
                existing_image = (current_state.values or {}).get("image")

                if image_base64 != existing_image:
                    # Persist image into the LangGraph state
                    with langfuse.start_as_current_observation(name="user-upload-image") as trace:
                        with propagate_attributes(session_id=thread_id):
                            rich_input = [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": "User uploaded an image manually via sidebar."},
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": image_data_uri
                                            }
                                        }
                                    ]
                                }
                            ]
                            
                            # Log the rich input
                            trace.update(input=rich_input)
                            
                            # Update the actual agent state
                            agent.update_state(base_config, {"image": image_base64})
                            
                            trace.update(output={"status": "success", "info": "Image persisted to graph state"})
                

                is_waiting = current_state.next and "wait_for_image" in current_state.next
                if is_waiting:
                    # Resume without additional user input
                    run_agent_sync(agent, langfuse, message=None, config=cb_config, thread_id=thread_id)
                    st.rerun()
            except Exception as e:
                st.sidebar.warning(f"Could not resume agent automatically: {e}")


def main():
    agent, builder = get_agent_and_builder()
    langfuse = get_client()
    langfuse.flush()
    # Ensure a per-session thread id for checkpointer
    if "thread_id" not in st.session_state:
        print("Initialize thread_id")
        st.session_state.thread_id = str(uuid.uuid4())

    # Render Graph
    try:
        st.header("Graph")
        png = builder.render_graph(agent)
        st.image(png)
    except Exception:
        pass

    # Show sidebar
    sidebar(agent, langfuse, st.session_state.thread_id)

    # Main chat area
    st.header("Chat")

    # 1. Show chat history
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    snapshot = agent.get_state(config)
    messages = []
    if snapshot.values:
        messages = snapshot.values.get("messages", [])
    render_chat_history(messages)

    # 2. Handle interrupt (waiting for image) — check checkpoint, not session state
    is_waiting_for_image = snapshot.next and "wait_for_image" in snapshot.next
    if is_waiting_for_image:
        st.warning("⏸️ Waiting for image upload. Please upload or take a photo in the sidebar above.")
        return  # Don't show chat input
    

    # 3. Chat input (only if not interrupted)
    prompt = "Enter a question about arithmetic or ask for hanzi classification"
    user_input = st.chat_input(prompt)

    if user_input:
        config = {"configurable": {"thread_id": st.session_state.thread_id}, "callbacks": [LANGFUSE_HANDLER]}

        # Run agent
        try:
            with st.spinner("Agent thinking..."):
                run_agent_sync(agent, langfuse, message=user_input, config=config, thread_id=st.session_state.thread_id)
        except Exception as e:
            st.error(f"Agent invocation failed: {e}")
            return

        st.rerun()  # Rerun to show updated conversation


if __name__ == "__main__":
    main()