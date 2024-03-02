import streamlit as st


def set_initial_state():
    ###########
    # General #
    ###########
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": "Welcome !",
            }
        ]


def set_page_config():
    st.set_page_config(
        page_title="AutoRAG",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get help': "https://github.com/Marker-Inc-Korea/AutoRAG/discussions",
            'Report a bug': "https://github.com/Marker-Inc-Korea/AutoRAG/issues",
        }
    )


def set_page_header():
    st.header("📚 AutoRAG", anchor=False)
    st.caption(
        "Input a question and get an answer from the given documents. "
    )
