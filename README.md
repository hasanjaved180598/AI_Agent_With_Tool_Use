# 🤖 AI Agent with Tool Use

An autonomous AI agent powered by **LLaMA3** that decides on its own which tools to use to answer any question — searching the web, performing calculations, looking up Wikipedia, or reading uploaded documents — using the **ReAct (Reasoning + Acting)** framework.

---

## 📌 Table of Contents

- [What Makes This an Agent, Not a Chatbot](#-what-makes-this-an-agent-not-a-chatbot)
- [Available Tools](#-available-tools)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [How ReAct Works](#-how-react-works)
- [Key Concepts](#-key-concepts)
- [Real Debugging Journey](#-real-debugging-journey)
- [Demo](#-demo)
- [Setup and Installation](#-setup-and-installation)
- [How to Run](#-how-to-run)
- [Project Workflow](#-project-workflow)

---

## 🎯 What Makes This an Agent, Not a Chatbot

| | Chatbot | AI Agent |
|---|---|---|
| Knowledge source | Training data only | Training data + real-time tools |
| Current information | ❌ No | ✅ Web search |
| Calculations | ❌ Unreliable | ✅ Guaranteed correct via calculator tool |
| Custom documents | ❌ No | ✅ Document reader |
| Decision process | Answers directly | Plans → Acts → Observes → Repeats → Answers |
| Transparency | Black box | Full reasoning trace visible |

A chatbot generates a response directly from what it already knows. An agent **reasons about what it needs**, picks the right tool, uses it, reads the result, and repeats this loop until it has enough information for a final answer.

---

## 🛠️ Available Tools

| Tool | Purpose | Example Question |
|------|---------|-----------------|
| 🔍 Web Search | Current events, recent news | "What is the latest AI news in Pakistan?" |
| 🧮 Calculator | Math and calculations | "What is 15% of 85,000?" |
| 📖 Wikipedia | Facts, history, concepts | "Who invented the telephone?" |
| 📄 Document Reader | Uploaded .txt file content | "What does the report say about revenue?" |

Each tool has a name and a description. The agent reads these descriptions and **decides on its own** which tool fits the question — it is never explicitly told which one to use.

---

## 📁 Project Structure

```
ai_agent/
├── src/
│   ├── tools.py            ← all four tool definitions
│   ├── agent.py            ← ReAct agent builder + prompt template
│   └── memory.py           ← conversation memory management
├── app/
│   └── app.py              ← Streamlit chat interface
├── notebook/
│   └── exploration.ipynb   ← tool testing notebook
└── requirements.txt
```

---

## 🛠 Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Language | Python 3.11 | Core language |
| LLM | LLaMA3 8B via Ollama | Reasoning and generation |
| Agent Framework | LangChain (ReAct) | Tool orchestration |
| Web Search | DuckDuckGo Search | Free, no API key |
| Encyclopedia | Wikipedia API | Factual lookups |
| Memory | ConversationBufferWindowMemory | Last 10 message pairs |
| Dashboard | Streamlit | Chat interface with reasoning trace |

---

## 🧠 How ReAct Works

For every question, the agent follows this loop:

```
Question: "What is 20% of the current Bitcoin price?"

Thought : I need the current Bitcoin price first, then calculate 20%
Action  : web_search
Input   : "Bitcoin price today USD"
Observation: "$67,432"

Thought : Now I can calculate 20% of 67432
Action  : calculator
Input   : "67432 * 0.20"
Observation: "Result: 13486.4"

Thought : I now know the final answer
Final Answer: 20% of the current Bitcoin price ($67,432) is $13,486.40
```

This **Thought → Action → Observation** cycle repeats until the agent has enough information. The entire reasoning process is visible in the app — nothing is hidden.

---

## 🧠 Key Concepts

### Sandboxed Calculator
LLMs are unreliable at exact arithmetic. The calculator tool guarantees correct results — but it cannot simply use Python's `eval()` directly since that would let arbitrary code execute. Protection is implemented in two layers:

```python
allowed = set('0123456789+-*/().**math. ')
if not all(c in allowed or c.isalpha() for c in expression):
    return "Error: Invalid characters in expression."

result = eval(expression, {"__builtins__": {}}, {"math": math})
```

A character whitelist blocks dangerous input before it reaches `eval()`, and stripping `__builtins__` removes access to `import`, `open`, `exec`, and every other risky built-in function — only the `math` module is explicitly allowed back in.

---

### Wikipedia with Automatic Fallback
Wikipedia's API occasionally returns empty responses on certain networks. The tool automatically falls back to DuckDuckGo search if Wikipedia fails or returns too little content:

```python
result = wiki.run(query)
if result and len(result.strip()) > 10:
    return result
return DuckDuckGoSearchRun().run(query)
```

---

### Lightweight Document Retrieval
The document reader uses keyword-overlap scoring to find relevant paragraphs — a simpler alternative to the FAISS + embeddings approach used in the RAG Chatbot project:

```python
query_words = set(query.lower().split())
score = len(query_words & para_words)  # set intersection
```

---

### Conversation Memory
`ConversationBufferWindowMemory(k=10)` keeps the last 10 message pairs and feeds them back into every prompt as `{chat_history}`, allowing the agent to remember context like your name or earlier questions.

---

### Safety Limits
`max_iterations=8` in the `AgentExecutor` caps how many tool-use cycles can happen before the agent is forced to stop — preventing infinite loops if the model gets confused.

---

## 🐛 Real Debugging Journey

This project surfaced genuine production issues with local LLM agents — documented here because they are valuable, real-world lessons.

**Issue 1 — Format breaking (`_Exception` errors)**
The original prompt had too many extra instructions, causing LLaMA3 to lose track of the required `Thought/Action/Action Input` format and producing `_Exception` errors at every step.
**Fix:** Simplified the prompt to the bare-minimum ReAct format LLaMA3 was actually trained on. Simpler prompts outperform elaborate ones with smaller local models.

**Issue 2 — Wrong tool selection causing hallucination**
With a document loaded, asking "When was the company founded?" caused the agent to search Wikipedia instead of reading the uploaded document — returning a completely unrelated company's founding date as a confident answer.
**Fix:** Added one explicit instruction to the prompt: *"If a document has been uploaded, always try document_reader FIRST."* — a pure prompt-engineering fix, no code logic changed.

**Issue 3 — Unnecessary tool chaining causing iteration limit**
After successfully reading "Total revenue was PKR 75 million" from a document, the agent kept trying to reformat the number using the calculator, failing repeatedly on text like "PKR" and "million", and exhausting all 8 iterations before giving a Final Answer.
**Fix:** Added a rule telling the agent to use document_reader's plain-text answer directly unless the question explicitly requires a calculation.

These fixes illustrate a key lesson: with local smaller models like LLaMA3, **prompt clarity matters more than prompt length** — and debugging agents requires reading the reasoning trace, not just the final output.

---

## 🎬 Demo

> 📹 **Watch the full demo video below.**

[![AI Agent Demo](https://img.shields.io/badge/▶_Watch_Demo-YouTube-red?logo=youtube&style=for-the-badge)](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)

### What the demo covers:
- Asking a calculator question and watching the agent reason through it step by step
- Asking a factual question with automatic Wikipedia/web search fallback
- Uploading a document and asking questions answered directly from its content
- Viewing the full Thought → Action → Observation trace for each answer
- Demonstrating conversation memory across multiple turns

---

## ⚙️ Setup and Installation

### Prerequisites
- Python 3.11
- Ollama installed ([download here](https://ollama.com/download))

### Steps

```bash
# 1. Pull LLaMA3 (one time only — ~4.7GB)
ollama pull llama3

# 2. Clone the repository
git clone https://github.com/hasanjaved180598/AI_Agent_With_Tool_Use.git
cd AI_Agent_With_Tool_Use

# 3. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux

# 4. Install dependencies
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## 🚀 How to Run

```bash
# Make sure Ollama is running (auto-starts on Windows after installation)
.venv\Scripts\python.exe -m streamlit run app/app.py
```

Dashboard opens at `http://localhost:8501`

**Try these questions to test each tool:**

```
"What is 15% of 85000?"                               → uses calculator
"Who invented the telephone?"                          → uses wikipedia
"What is the latest AI news in Pakistan?"              → uses web_search
Upload a .txt file, then ask about its content         → uses document_reader
"My name is Hasan. What is my name?"  (two turns)     → tests memory
```

---

## 🔄 Project Workflow

```
User Question
    │
    ▼
agent.py (AgentExecutor)
  ├── Inserts tools + chat_history + question into prompt
  ├── Sends prompt to LLaMA3 via Ollama
  └── LLaMA3 responds: Thought → Action → Action Input
    │
    ▼
LangChain routes to the chosen tool in tools.py
  ├── web_search       → DuckDuckGoSearchRun
  ├── calculator       → safe_calculate() (sandboxed eval)
  ├── wikipedia        → WikipediaAPIWrapper + DuckDuckGo fallback
  └── document_reader  → keyword-overlap paragraph scoring
    │
    ▼
Observation (tool result) fed back into the prompt
    │
    ▼
LLaMA3 decides:
  need more info? → repeat Thought/Action loop
  have enough?    → Final Answer
    │
    ▼
memory.py saves the exchange (keeps last 10 pairs)
    │
    ▼
app.py displays:
  ├── Final answer
  ├── Tool usage icons
  └── Expandable reasoning trace (Thought/Action/Observation per step)
```

---

## ⚠️ Known Limitations

- DuckDuckGo's free search API has rate limits and occasional timeouts on certain networks. A production system would replace this with a paid search API (Tavily, SerpAPI, or Google Custom Search) for reliability.
- LLaMA3 8B occasionally deviates from the exact ReAct format — `handle_parsing_errors=True` and a simplified prompt mitigate this, but it is a known characteristic of smaller local models.
- Document reader uses simple keyword matching rather than semantic search — effective for short files, less so for large or dense documents where a full RAG pipeline would be more appropriate.

---

## 📄 License

This project is licensed under the MIT License.

---

*Because a great assistant doesn't just answer questions — it knows how to find the answers. 🤖🔍*
