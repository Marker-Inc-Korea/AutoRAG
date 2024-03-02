import click
import streamlit as st

from autorag.deploy import Runner


def get_runner(yaml_path: str, project_dir: str):
    return Runner.from_yaml(yaml_path, project_dir=project_dir)


def set_initial_state():
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


def chat_box(runner: Runner):
    if query := st.chat_input("How can I help?"):
        # Add the user input to messages state
        st.session_state["messages"].append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Generate llama-index stream with user input
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                response = st.write(runner.run(query))

        # Add the final response to messages state
        st.session_state["messages"].append({"role": "assistant", "content": response})


@click.command()
@click.option("--yaml_path", type=str, help="Path to the YAML file.")
@click.option("--project_dir", type=str, help="Path to the project directory.")
def run_web_server(yaml_path, project_dir):
    import nest_asyncio
    nest_asyncio.apply()
    runner = get_runner(yaml_path, project_dir)
    set_initial_state()
    set_page_config()
    set_page_header()
    chat_box(runner)


if __name__ == "__main__":
    run_web_server()
