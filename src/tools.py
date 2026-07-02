
import math
import re
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper


def get_web_search_tool() -> Tool:

    search = DuckDuckGoSearchRun()

    return Tool(
        name='web_search',
        description=(
            'Search the internet for current information, recent news, '
            'or any topic that requires up-to-date data. '
            'Use this when you need information about recent events, '
            'current prices, latest developments, or anything that '
            'might have changed recently. '
            'Input should be a search query string.'
        ),
        func=search.run
    )


def safe_calculate(expression: str) -> str:

    try:
        expression = expression.strip()

        expression = expression.replace('^', '**')
        expression = expression.replace('×', '*')
        expression = expression.replace('÷', '/')
        expression = expression.replace('sqrt', 'math.sqrt')
        expression = expression.replace('sin',  'math.sin')
        expression = expression.replace('cos',  'math.cos')
        expression = expression.replace('tan',  'math.tan')
        expression = expression.replace('log',  'math.log')
        expression = expression.replace('pi',   'math.pi')
        expression = expression.replace('e',    'math.e')

        allowed = set('0123456789+-*/().**math. ')
        if not all(c in allowed or c.isalpha() for c in expression):
            return "Error: Invalid characters in expression."

        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return f"Result: {result}"

    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as e:
        return f"Error calculating: {str(e)}"


def get_calculator_tool() -> Tool:

    return Tool(
        name='calculator',
        description=(
            'Perform mathematical calculations. '
            'Use this for arithmetic, percentages, algebra, '
            'square roots, and any numerical computation. '
            'Input should be a mathematical expression like '
            '"15 * 0.15" or "sqrt(144)" or "2 ** 10". '
            'Always use this tool instead of computing in your head.'
        ),
        func=safe_calculate
    )


def get_wikipedia_tool() -> Tool:

    def wikipedia_search(query: str) -> str:
        try:
            wiki = WikipediaAPIWrapper(
                top_k_results=1,
                doc_content_chars_max=800
            )
            result = wiki.run(query)
            if result and len(result.strip()) > 10:
                return result
            search = DuckDuckGoSearchRun()
            return search.run(query)
        except Exception:
            try:
                search = DuckDuckGoSearchRun()
                return search.run(query)
            except Exception as e:
                return f"Could not retrieve information: {str(e)}"

    return Tool(
        name='wikipedia',
        description=(
            'Look up factual information about history, science, geography, '
            'famous people, concepts, and definitions. '
            'Do NOT use this for current events or recent news — use web_search instead. '
            'Input should be a search query.'
        ),
        func=wikipedia_search
    )



_loaded_document = {'content': None, 'name': None}


def load_document(content: str, name: str):
    _loaded_document['content'] = content
    _loaded_document['name']    = name


def clear_document():
    _loaded_document['content'] = None
    _loaded_document['name']    = None


def read_document(query: str) -> str:

    if not _loaded_document['content']:
        return "No document is currently loaded. Ask the user to upload a text file first."

    content  = _loaded_document['content']
    doc_name = _loaded_document['name']

    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

    if not paragraphs:
        paragraphs = [p.strip() for p in content.split('\n') if p.strip()]

    query_words = set(query.lower().split())
    scored_paragraphs = []

    for para in paragraphs:
        para_words = set(para.lower().split())
        score = len(query_words & para_words)
        if score > 0:
            scored_paragraphs.append((score, para))

    scored_paragraphs.sort(reverse=True, key=lambda x: x[0])

    if scored_paragraphs:
        relevant = '\n\n'.join([p for _, p in scored_paragraphs[:3]])
        return f"From '{doc_name}':\n\n{relevant}"
    else:
        return f"From '{doc_name}' (no specific match found):\n\n{content[:500]}..."


def get_document_reader_tool() -> Tool:

    return Tool(
        name='document_reader',
        description=(
            'ALWAYS use this tool FIRST for ANY question, before any other tool. '
            'This searches an uploaded document for relevant information. '
            'If the user has uploaded a document, the answer is very likely in it. '
            'Use this for questions about companies, people, dates, numbers, or '
            'facts that could be in the uploaded file — even if not explicitly '
            'mentioned. Input should be the topic or question to search for.'
        ),
        func=read_document
    )


def get_all_tools() -> list:
    return [
        get_web_search_tool(),
        get_calculator_tool(),
        get_wikipedia_tool(),
        get_document_reader_tool(),
    ]