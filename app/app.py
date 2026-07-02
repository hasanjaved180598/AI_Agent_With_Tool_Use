
import os
import sys
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.agent import build_agent, run_agent
from src.memory import create_memory, clear_memory
from src.tools import load_document, clear_document

st.set_page_config(
    page_title='AI Agent',
    page_icon='🤖',
    layout='wide'
)

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'agent' not in st.session_state:
    st.session_state.agent  = None
    st.session_state.memory = None

if 'document_loaded' not in st.session_state:
    st.session_state.document_loaded = False
    st.session_state.document_name   = None


@st.cache_resource
def get_agent():

    memory         = create_memory()
    agent_executor, memory = build_agent(memory)
    return agent_executor, memory


agent_executor, memory = get_agent()


st.title('🤖 AI Agent with Tool Use')
st.markdown(
    'Powered by **LLaMA3** via Ollama. This agent can **search the web**, '
    '**perform calculations**, **look up Wikipedia**, and **read uploaded documents** — '
    'it decides which tool to use on its own.'
)
st.divider()

with st.sidebar:
    st.header('🛠️ Available Tools')

    st.markdown("""
    | Tool | When Used |
    |------|-----------|
    | 🔍 Web Search | Current events, recent news |
    | 🧮 Calculator | Math and calculations |
    | 📖 Wikipedia | Facts, history, concepts |
    | 📄 Document Reader | Uploaded file content |
    """)

    st.divider()
    st.header('📄 Upload Document')

    uploaded_file = st.file_uploader(
        'Upload a text file (.txt)',
        type=['txt'],
        help='Upload a text file to ask questions about it'
    )

    if uploaded_file:
        if st.session_state.document_name != uploaded_file.name:
            content = uploaded_file.read().decode('utf-8')
            load_document(content, uploaded_file.name)
            st.session_state.document_loaded = True
            st.session_state.document_name   = uploaded_file.name
            st.success(f'✅ Loaded: {uploaded_file.name}')
    
    if st.session_state.document_loaded:
        st.info(f'📄 Active: {st.session_state.document_name}')
        if st.button('🗑️ Remove Document', use_container_width=True):
            clear_document()
            st.session_state.document_loaded = False
            st.session_state.document_name   = None
            st.rerun()

    st.divider()
    st.header('💬 Example Questions')
    st.markdown("""
    **Web Search:**
    - "What is the latest news about AI in Pakistan?"
    - "What is the current USD to PKR exchange rate?"

    **Calculator:**
    - "What is 15% of 85000?"
    - "Calculate compound interest on 100000 at 12% for 3 years"

    **Wikipedia:**
    - "Who invented the telephone?"
    - "What is machine learning?"

    **Document:**
    - Upload a .txt file then ask:
    - "What does the document say about X?"
    """)

    st.divider()
    if st.button('🗑️ Clear Chat History', use_container_width=True):
        st.session_state.messages = []
        clear_memory(memory)
        st.rerun()


for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

        if message['role'] == 'assistant' and message.get('steps'):
            with st.expander('🔍 View Agent Reasoning'):
                for i, step in enumerate(message['steps']):
                    action, observation = step
                    st.markdown(f"**Step {i+1}:**")
                    st.markdown(f"🛠️ **Tool:** `{action.tool}`")
                    st.markdown(f"📥 **Input:** {action.tool_input}")
                    st.markdown(f"📤 **Result:** {observation[:300]}{'...' if len(str(observation)) > 300 else ''}")
                    if i < len(message['steps']) - 1:
                        st.divider()

if question := st.chat_input('Ask me anything — I will use the right tool automatically ...'):

    st.session_state.messages.append({'role': 'user', 'content': question})
    with st.chat_message('user'):
        st.markdown(question)

    with st.chat_message('assistant'):
        with st.spinner('Thinking and using tools ...'):
            result = run_agent(agent_executor, question)

        answer = result['output']
        steps  = result['intermediate_steps']

        st.markdown(answer)

        if steps:
            tools_used = [step[0].tool for step in steps]
            tool_icons = {
                'web_search':      '🔍',
                'calculator':      '🧮',
                'wikipedia':       '📖',
                'document_reader': '📄'
            }
            icons = ' '.join([tool_icons.get(t, '🛠️') for t in tools_used])
            st.caption(f"Tools used: {icons} {', '.join(tools_used)}")

            with st.expander('🔍 View Agent Reasoning'):
                for i, step in enumerate(steps):
                    action, observation = step
                    st.markdown(f"**Step {i+1}:**")
                    st.markdown(f"🛠️ **Tool:** `{action.tool}`")
                    st.markdown(f"📥 **Input:** {action.tool_input}")
                    st.markdown(f"📤 **Result:** {str(observation)[:300]}{'...' if len(str(observation)) > 300 else ''}")
                    if i < len(steps) - 1:
                        st.divider()

    # Save to history
    st.session_state.messages.append({
        'role':    'assistant',
        'content': answer,
        'steps':   steps
    })
