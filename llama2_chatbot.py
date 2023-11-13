"""
LLaMA 2 Chatbot app
======================

This is a Streamlit chatbot app with LLaMA2 that includes session chat history and an option to select multiple LLM
API endpoints on Replicate. The 7B and 13B models run on Replicate on one A100 40Gb. The 70B runs in one A100 80Gb. The weights have been tensorized.

Author: Marco Mascorro (@mascobot.com)
Created: July 2023
Version: 0.9.0 (Experimental)
Status: Development
Python version: 3.9.15
a16z-infra
"""
#External libraries:
import streamlit as st
#import replicate
#from dotenv import load_dotenv
#load_dotenv()
import os
import openai
from openai.error import OpenAIError
from openai.error import AuthenticationError
from langchain.llms import OpenAI
from OpenAICustom import OpenAICustom

###Initial UI configuration:###
st.set_page_config(page_title="LLaMA2 Chatbot by a16z-infra", page_icon="🦙", layout="wide")

#Modelendpoint
openai.api_key = "Empty"
openai.api_base = "http://91.92.117.167:8000/v1"
model = "code-llama2-34b"


def build_llama2_prompt(messages, k=3):
    startPrompt = "<s>[INST] "
    endPrompt = " [/INST]"
    conversation = []

    startAppend = len(messages)-k*2-1
    prePromptMessage = st.session_state['pre_prompt']

    conversation.append(f"<<SYS>>\n{prePromptMessage.strip()}\n<</SYS>>\n\n")

    for index, message in enumerate(messages):                    
        if message["role"] == "user" and index >= startAppend:
            conversation.append(message["content"].strip())
        elif index >= startAppend:
            conversation.append(f" [/INST] {message['content'].strip()} </s><s>[INST] ")

    return startPrompt + "".join(conversation) + endPrompt
  


# reduce font sizes for input text boxes
custom_css = """
    <style>
        .stTextArea textarea {font-size: 13px;}
        div[data-baseweb="select"] > div {font-size: 13px !important;}
    </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

#Left sidebar menu
st.sidebar.header("LLaMA2 Chatbot")

#Set config for a cleaner menu, footer & background:
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

PRE_PROMPT=""

#container for the chat history
response_container = st.container()
#container for the user's text input
container = st.container()
#Set up/Initialize Session State variables:
if 'chat_dialogue' not in st.session_state:
    st.session_state['chat_dialogue'] = []
if 'temperature' not in st.session_state:
    st.session_state['temperature'] = 0.1
if 'top_p' not in st.session_state:
    st.session_state['top_p'] = 0.9
if 'max_seq_len' not in st.session_state:
    st.session_state['max_seq_len'] = 512
if 'pre_prompt' not in st.session_state:
    st.session_state['pre_prompt'] = PRE_PROMPT
if 'string_dialogue' not in st.session_state:
    st.session_state['string_dialogue'] = ''

#Dropdown menu to select the model edpoint:
selected_option = st.sidebar.selectbox('Choose a LLaMA2 model:', ['code-llama2-34b'], key='model')
    
#Model hyper parameters:
st.session_state['temperature'] = st.sidebar.slider('Temperature:', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
st.session_state['top_p'] = st.sidebar.slider('Top P:', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
st.session_state['max_seq_len'] = st.sidebar.slider('Max Sequence Length:', min_value=64, max_value=16384, value=4096, step=8)

st.session_state.LLM = OpenAI(temperature=st.session_state['temperature'] , openai_api_base=openai.api_base, model=model, openai_api_key=openai.api_key, max_tokens=st.session_state['max_seq_len'])



NEW_P = st.sidebar.text_area('Prompt before the chat starts. Edit here if desired:', PRE_PROMPT, height=60)
if NEW_P != PRE_PROMPT and NEW_P != "" and NEW_P != None:
    st.session_state['pre_prompt'] = NEW_P + "\n\n"
else:
    st.session_state['pre_prompt'] = PRE_PROMPT

btn_col1, btn_col2 = st.sidebar.columns(2)

# Add the "Clear Chat History" button to the sidebar
def clear_history():
    st.session_state['chat_dialogue'] = []
clear_chat_history_button = btn_col1.button("Clear History",
                                        use_container_width=True,
                                        on_click=clear_history)

    
# Display chat messages from history on app rerun
for message in st.session_state.chat_dialogue:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Type your question here to talk to LLaMA2"):
    # Add user message to chat history
    st.session_state.chat_dialogue.append({"role": "user", "content": prompt})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    #model
    prompt = build_llama2_prompt(st.session_state.chat_dialogue, k=3)
    print(prompt)
    
    answer =  st.session_state.LLM.predict(prompt)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        string_dialogue = st.session_state['pre_prompt']
        for dict_message in st.session_state.chat_dialogue:
            if dict_message["role"] == "user":
                string_dialogue = string_dialogue + "User: " + dict_message["content"] + "\n\n"
            else:
                string_dialogue = string_dialogue + "Assistant: " + dict_message["content"] + "\n\n"
        print (string_dialogue)
        #output = debounce_replicate_run(st.session_state['llm'], string_dialogue + "Assistant: ",  st.session_state['max_seq_len'], st.session_state['temperature'], st.session_state['top_p'], REPLICATE_API_TOKEN)
        #for item in output:
        #    full_response += item
        #    message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(answer)
    # Add assistant response to chat history
    st.session_state.chat_dialogue.append({"role": "assistant", "content": answer})


