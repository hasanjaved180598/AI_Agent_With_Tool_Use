
from langchain.memory import ConversationBufferWindowMemory
from typing import List, Dict


def create_memory() -> ConversationBufferWindowMemory:

    return ConversationBufferWindowMemory(
        k=10,
        memory_key='chat_history',
        return_messages=True,
        input_key='input',
        output_key='output'
    )


def format_chat_history(memory: ConversationBufferWindowMemory) -> List[Dict]:

    messages = []
    history  = memory.chat_memory.messages

    for msg in history:
        role    = 'human' if msg.__class__.__name__ == 'HumanMessage' else 'ai'
        messages.append({'role': role, 'content': msg.content})

    return messages


def clear_memory(memory: ConversationBufferWindowMemory):
    memory.clear()
