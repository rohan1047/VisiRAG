import streamlit as st
import os
import tempfile
from backend import get_rag_engine

st.title("PDF RAG Chat - Test Interface")

# Initialize session state
if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

# File upload section
st.header("Upload PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        pdf_path = tmp_file.name
    
    # Initialize RAG engine if not already done
    if st.session_state.rag_engine is None:
        st.write("Processing PDF...")
        st.session_state.rag_engine = get_rag_engine(pdf_path)
        if st.session_state.rag_engine:
            st.success("PDF processed successfully!")
        else:
            st.error("Failed to process PDF")

# Chat interface
if st.session_state.rag_engine:
    st.header("Chat")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["content"])
            else:
                # Display assistant response with inline images
                processed_response = message["content"]
                if "images" in message and message["images"]:
                    for img_info in message["images"]:
                        if img_info["path"] and os.path.exists(img_info["path"]) and img_info.get("exists"):
                            image_placeholder = f"[IMAGE:{img_info['ref']}]"
                            if image_placeholder in processed_response:
                                parts = processed_response.split(image_placeholder, 1)
                                st.write(parts[0])
                                st.image(img_info["path"], caption=f"Image: {img_info['ref']}", use_container_width=True)
                                processed_response = parts[1] if len(parts) > 1 else ""
                    
                    if processed_response:
                        st.write(processed_response)
                else:
                    st.write(processed_response)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the PDF"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get response from RAG engine
        with st.chat_message("assistant"):
            result = st.session_state.rag_engine.query(prompt)
            
            # Display response with inline images
            processed_response = result["response"]
            if result["images"]:
                for img_info in result["images"]:
                    if img_info["path"] and os.path.exists(img_info["path"]) and img_info.get("exists"):
                        image_placeholder = f"[IMAGE:{img_info['ref']}]"
                        if image_placeholder in processed_response:
                            parts = processed_response.split(image_placeholder, 1)
                            st.write(parts[0])
                            st.image(img_info["path"], caption=f"Image: {img_info['ref']}", use_container_width=True)
                            processed_response = parts[1] if len(parts) > 1 else ""
                
                if processed_response:
                    st.write(processed_response)
            else:
                st.write(result["response"])
            
            # Add assistant message to history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": result["response"],
                "images": result.get("images", [])
            })

# Reset button
if st.button("Reset Chat"):
    st.session_state.messages = []
    if st.session_state.rag_engine:
        st.session_state.rag_engine.clear_memory()
    st.rerun()

# Clear all button  
if st.button("Clear All"):
    st.session_state.rag_engine = None
    st.session_state.messages = []
    st.rerun()