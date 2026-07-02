
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.llms import Ollama
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from typing import Optional

from src.tools import get_all_tools
from src.memory import create_memory

LLM_MODEL = 'llama3'


def get_llm() -> Ollama:
        return Ollama(
        model=LLM_MODEL,
        temperature=0
    )


def build_prompt() -> PromptTemplate:

    template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

IMPORTANT: If a document has been uploaded, always try document_reader FIRST before using wikipedia or web_search. Only use wikipedia or web_search if document_reader says no document is loaded or the answer is not found in it. If document_reader already gives you the answer in plain text (like a number, date, or name), use that answer directly — do NOT use the calculator to reformat or convert it unless the question explicitly asks for a calculation.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Previous conversation:
{chat_history}

Question: {input}
Thought:{agent_scratchpad}"""

    return PromptTemplate(
        template=template,
        input_variables=[
            'tools', 'tool_names', 'chat_history', 'input', 'agent_scratchpad'
        ]
    )


def build_agent(memory: Optional[ConversationBufferWindowMemory] = None):

    llm    = get_llm()
    tools  = get_all_tools()
    prompt = build_prompt()

    if memory is None:
        memory = create_memory()

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=8,
        return_intermediate_steps=True
    )

    return agent_executor, memory


def run_agent(agent_executor, question: str) -> dict:

    try:
        result = agent_executor.invoke({'input': question})
        return {
            'output':             result.get('output', 'No answer generated.'),
            'intermediate_steps': result.get('intermediate_steps', []),
            'error':              None
        }
    except Exception as e:
        error_msg = str(e)
        if 'Could not parse LLM output' in error_msg:

            import re
            match = re.search(r'`(.*?)`', error_msg, re.DOTALL)
            answer = match.group(1) if match else "I encountered a formatting issue. Please try rephrasing your question."
            return {
                'output':             answer,
                'intermediate_steps': [],
                'error':              None
            }
        return {
            'output':             f"Error: {error_msg}",
            'intermediate_steps': [],
            'error':              error_msg
        }