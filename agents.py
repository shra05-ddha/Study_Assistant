# agents.py
from typing import Optional
from threading import Lock

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from llm_provider import get_llm

# A small process-wide lock to ensure pipeline.invoke() runs serially
# (prevents overlapping LLM calls from different agents/process threads)
_pipeline_lock = Lock()

# -------------------------
# 1. EXPLANATION AGENT
# -------------------------
explain_template = """
You are a helpful study assistant. Explain the following concept in simple clear language.
Use examples and analogies suitable for an undergraduate student.

QUESTION:
{question}

Additional Context (if available):
{context}

Provide a clear, easy explanation.
"""

explain_prompt = PromptTemplate(
    template=explain_template,
    input_variables=["question", "context"]
)


def explanation_agent(question: str, context: str = "") -> str:
    """Return a plain string explanation for `question` using LCEL pipeline."""
    llm = get_llm()
    pipeline = explain_prompt | llm | StrOutputParser()
    try:
        # ensure only one pipeline invocation is active at a time in this process
        with _pipeline_lock:
            result = pipeline.invoke({"question": question, "context": context})
        return str(result)
    except Exception as e:
        return f"[Error from explanation_agent] {e}"


# -------------------------
# 2. SUMMARIZATION AGENT
# -------------------------
summary_template = """
Summarize the following notes into short bullet points. Include key topics,
definitions, formulas, and important ideas.

NOTES:
{notes}
"""

summary_prompt = PromptTemplate(
    template=summary_template,
    input_variables=["notes"]
)


def summarization_agent(notes_text: str) -> str:
    llm = get_llm()
    pipeline = summary_prompt | llm | StrOutputParser()
    try:
        with _pipeline_lock:
            result = pipeline.invoke({"notes": notes_text})
        return str(result)
    except Exception as e:
        return f"[Error from summarization_agent] {e}"


# -------------------------
# 3. QUIZ GENERATION AGENT
# -------------------------
quiz_template = """
Generate {num} MCQ questions based on the content below.

CONTENT:
{content}

Difficulty: {difficulty}

Each question must include:
- 4 options (A, B, C, D)
- Correct answer
- A brief explanation

Return questions in a clean exam-ready format.
"""

quiz_prompt = PromptTemplate(
    template=quiz_template,
    input_variables=["num", "content", "difficulty"]
)


def quiz_agent(content: str, num: int, difficulty: str) -> str:
    llm = get_llm()
    pipeline = quiz_prompt | llm | StrOutputParser()
    try:
        with _pipeline_lock:
            result = pipeline.invoke({
                "num": str(num),
                "content": content,
                "difficulty": difficulty,
            })
        return str(result)
    except Exception as e:
        return f"[Error from quiz_agent] {e}"


# -------------------------
# 4. RAG ANSWERING AGENT
# -------------------------
rag_template = """
You are an AI tutor. Use ONLY the following notes to answer the question.

NOTES:
{context}

QUESTION:
{question}

Give a clear and accurate answer, cite the note line(s) where appropriate.
"""

rag_prompt = PromptTemplate(
    template=rag_template,
    input_variables=["context", "question"]
)


def rag_answer_agent(question: str, context: str) -> str:
    llm = get_llm()
    pipeline = rag_prompt | llm | StrOutputParser()
    try:
        with _pipeline_lock:
            result = pipeline.invoke({"context": context, "question": question})
        return str(result)
    except Exception as e:
        return f"[Error from rag_answer_agent] {e}"
