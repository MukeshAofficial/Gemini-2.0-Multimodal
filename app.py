import streamlit as st
from google import genai
from google.genai import types
import PIL.Image
from io import BytesIO

# Configure Streamlit page
st.set_page_config(page_title="Gemini Multimodal Chat", page_icon=":rocket:", layout="wide")

# --- Helper Functions ---
def get_search_results(api_key, search_query, model_name):
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=search_query,
            config=types.GenerateContentConfig(
                tools=[types.Tool(
                    google_search=types.GoogleSearchRetrieval()
                )]
            )
        )
        return response
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

def get_image_analysis(api_key, image_data, prompt_text, model_name):
    try:
        client = genai.Client(api_key=api_key)
        image = PIL.Image.open(BytesIO(image_data))
        contents = [prompt_text, image] if prompt_text else [image]
        response = client.models.generate_content(
            model=model_name,
            contents=contents
        )
        return response
    except Exception as e:
        st.error(f"An error occurred during image analysis: {e}")
        return None

def get_chat_response(api_key, prompt, model_name):
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.candidates[0].content.parts[0].text if response.candidates else None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

def display_chat_message(role, content, avatar=None):
    with st.chat_message(role, avatar=avatar):
        st.write(content)

def clear_chat_history():
    st.session_state.chat_history = []
    st.session_state.current_image = None

# --- Initialize Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "mode" not in st.session_state:
    st.session_state.mode = "chat"  # Default mode is chat

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Google API Key", type="password")
    model_options = {
        "gemini-2.0-flash": "gemini-2.0-flash",
        "gemini-2.0-flash-lite-preview-02-05": "gemini-2.0-flash-lite-preview-02-05",
        "gemini-2.0-pro-exp-02-05": "gemini-2.0-pro-exp-02-05",
    }
    model_name = st.selectbox("Select Model", options=list(model_options.keys()), format_func=lambda x: model_options[x])
    
    st.write("---")
    if st.button("Image Chat 🖼️", use_container_width=True):
        st.session_state.mode = "image"
        st.rerun()
    if st.button("Google Search 🔍", use_container_width=True):
        st.session_state.mode = "search"
        st.rerun()
    
    st.write("---")
    if st.button("Start New Chat 💬", use_container_width=True):
        clear_chat_history()
        st.session_state.mode = "chat"
        st.rerun()

# --- Main App Interface ---
st.title("✨ Gemini Multimodal Chat 🚀")
st.write("❤️ Built by [Build Fast with AI](https://buildfastwithai.com/genai-course)")

if st.session_state.mode == "image":
    # Show current image at the top if it exists
    if st.session_state.current_image:
        st.image(PIL.Image.open(BytesIO(st.session_state.current_image)), use_column_width=True)
    
    # Display chat history
    for message in st.session_state.chat_history:
        if isinstance(message["content"], tuple):
            display_chat_message("user", f"Asked: '{message['content'][2]}'", avatar="👤")
        elif message["role"] == "assistant":
            display_chat_message("assistant", message["content"], avatar="🤖")
    
    # Image upload and prompt input at the bottom
    with st.container():
        st.write("---")
        if not st.session_state.current_image:
            uploaded_file = st.file_uploader("Upload an image for analysis", type=["png", "jpg", "jpeg"])
            if uploaded_file:
                st.session_state.current_image = uploaded_file.read()
                st.rerun()
        
        if st.session_state.current_image:
            prompt = st.chat_input("Ask about the image...")
            
            if prompt and api_key:
                st.session_state.chat_history.append({"role": "user", "content": ("image", st.session_state.current_image, prompt)})
                response = get_image_analysis(api_key, st.session_state.current_image, prompt, model_name)
                if response and response.candidates:
                    response_text = response.candidates[0].content.parts[0].text
                    st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                st.rerun()

elif st.session_state.mode == "search":
    # Display chat history
    for message in st.session_state.chat_history:
        display_chat_message(message["role"], message["content"], 
                           avatar="👤" if message["role"] == "user" else "🤖")
        if message["role"] == "assistant" and "metadata" in message:
            with st.expander("Show Sources"):
                if message["metadata"]["sources"]:
                    for source in message["metadata"]["sources"]:
                        st.write(f"- [{source['title']}]({source['url']})")
                else:
                    st.write("No sources available.")
    
    # Search input at the bottom
    prompt = st.chat_input("Search with Google...")
    
    if prompt and api_key:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.spinner("Searching with Google..."):
            response = get_search_results(api_key, prompt, model_name)
            if response and response.candidates:
                response_text = response.candidates[0].content.parts[0].text
                sources = []
                if hasattr(response.candidates[0], 'grounding_metadata') and response.candidates[0].grounding_metadata:
                    for chunk in response.candidates[0].grounding_metadata.grounding_chunks:
                        if chunk.web and chunk.web.uri:
                            sources.append({
                                'title': chunk.web.title,
                                'url': chunk.web.uri
                            })
                
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response_text,
                    "metadata": {"sources": sources}
                })
                st.rerun()

else:  # Default chat mode
    # Display chat history
    for message in st.session_state.chat_history:
        display_chat_message(message["role"], message["content"], 
                           avatar="👤" if message["role"] == "user" else "🤖")
    
    # Chat input at the bottom
    prompt = st.chat_input("Chat with Gemini...")
    
    if prompt and api_key:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        response = get_chat_response(api_key, prompt, model_name)
        if response:
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()
