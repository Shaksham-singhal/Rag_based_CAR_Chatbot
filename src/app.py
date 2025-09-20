import streamlit as st
from llm_rag import car_rag_pipeline

# Initialize session state
if "chats" not in st.session_state:
    st.session_state.chats = []  # list of conversations
if "current_chat" not in st.session_state:
    st.session_state.current_chat = []
if "active_chat_index" not in st.session_state:
    st.session_state.active_chat_index = None

def new_chat():
    # Just reset without appending to avoid duplicates
    st.session_state.current_chat = []
    st.session_state.active_chat_index = None

def delete_chat(idx):
    if 0 <= idx < len(st.session_state.chats):
        del st.session_state.chats[idx]
        st.session_state.active_chat_index = None
        st.session_state.current_chat = []

def main():
    st.set_page_config(page_title="Car RAG Chatbot", layout="wide")

    # Sidebar with chat history
    with st.sidebar:
        st.title("💬 Chats")
        if st.button("➕ New Chat"):
            new_chat()

        for i, chat in enumerate(st.session_state.chats):
            # Use first user query as title
            title = chat[0][1][:30] + "..." if chat else f"Chat {i+1}"
            cols = st.columns([4, 1])
            with cols[0]:
                if st.button(title, key=f"chat_{i}"):
                    st.session_state.active_chat_index = i
                    st.session_state.current_chat = chat.copy()
            with cols[1]:
                if st.button("🗑️", key=f"del_{i}"):
                    delete_chat(i)
                    st.rerun()

    st.title("🚗 Car RAG Chatbot")

    # Display conversation with single container bubbles
    for role, text in st.session_state.current_chat:
        if role == "user":
            st.markdown(
                f"""
                <div style='display:flex; justify-content:flex-end; margin:6px 0;'>
                    <div style='background-color:#0B93F6; color:white;
                                padding:10px 14px; border-radius:12px; 
                                max-width:70%; word-wrap:break-word;
                                text-align:left;'>
                        <b>You:</b> {text}
                    </div>
                </div>
                """, unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style='display:flex; justify-content:flex-start; margin:6px 0;'>
                    <div style='background-color:#262626; color:white;
                                padding:10px 14px; border-radius:12px; 
                                max-width:70%; word-wrap:break-word;
                                text-align:left;'>
                        <b>Bot:</b> {text}
                    </div>
                </div>
                """, unsafe_allow_html=True
            )

    # Input area at bottom
    query = st.chat_input("Ask something about cars...")
    if query:
        # Save user message
        st.session_state.current_chat.append(("user", query))

        # Generate bot response
        answer = car_rag_pipeline(query)
        st.session_state.current_chat.append(("bot", answer))

        # If it’s a new chat, save as active
        if st.session_state.active_chat_index is None:
            st.session_state.active_chat_index = len(st.session_state.chats)
            st.session_state.chats.append(st.session_state.current_chat)
        else:
            st.session_state.chats[st.session_state.active_chat_index] = st.session_state.current_chat

        st.rerun()

if __name__ == "__main__":
    main()
