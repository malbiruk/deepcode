"""
Streamlit chat with DeepSeek
"""

import re

import ollama
import streamlit as st
from langchain_ollama import OllamaLLM
from PIL import Image


def customize_page_appearance() -> None:
    """
    remove streamlit"s red bar and "deploy" button at the top
    """
    st.markdown("""
        <style>
            [data-testid="stDecoration"] {
                display: none;
            }
            .stDeployButton {
                visibility: hidden;
            }
        </style>
        """, unsafe_allow_html=True)


chat_avatars = {
    "user": ":material/arrow_forward_ios:",
    "human": ":material/arrow_forward_ios:",
    "assistant": ":material/psychology:",
    "ai": ":material/psychology:",
}


def extract_think_content(response):
    think_pattern = r"<think>(.*?)</think>"
    think_match = re.search(think_pattern, response, re.DOTALL)

    if think_match:
        think_content = think_match.group(1).strip()
        main_response = re.sub(think_pattern, "", response, flags=re.DOTALL).strip()
        return think_content, main_response
    return None, response


def generate_deepseek_response(llm):
    string_dialogue = "".join(
        f"{msg['role']}: {msg['content']}\n\n"
        for msg in st.session_state.messages
    )
    return llm.stream(string_dialogue)


def process_stream(response_stream, response_list):
    for chunk_ in response_stream:
        chunk = chunk_.replace("<think>", "")
        chunk = chunk.replace("</think>", "\n\n**Answer:**")
        response_list.append(chunk_)
        yield chunk


def main():
    im = Image.open("/home/klim/Documents/work/scripts/deepcode/icon.png")
    st.set_page_config(page_title="Ollama Chat",
                       page_icon=im)
    customize_page_appearance()

    model = st.selectbox("Model", [
        model.model for model in ollama.list()["models"]
    ])

    llm = OllamaLLM(
        model=model,
        temperature=0,
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"], avatar=chat_avatars[msg["role"]]).write(msg["content"])

    if prompt := st.chat_input("What is up?"):
        st.chat_message("user", avatar=chat_avatars["user"]).write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant", avatar=chat_avatars["assistant"]):
                response_generator = generate_deepseek_response(llm)
                if "-r1" in model:
                    status_container = st.status("Thinking ...", expanded=True)
                    response_list = []
                    with status_container:
                        st.write_stream(process_stream(response_generator, response_list))
                        think_content, main_response = extract_think_content("".join(response_list))

                    status_container.update(label="Thoughts", state="complete", expanded=False)
                    st.markdown(main_response)
                else:
                    main_response = st.write_stream(response_generator)
                st.session_state.messages.append(
                    {"role": "assistant", "content": main_response})


if __name__ == "__main__":
    main()
