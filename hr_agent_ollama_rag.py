"""
AI HR Recruitment Assistant — Local RAG + Agent implementation using Ollama
=============================================================================

Runs entirely on your machine against a local Ollama server (no cloud API).

Requires Ollama running locally with two pulled models:
    ollama pull nomic-embed-text     # embedding model  -> powers RAG retrieval
    ollama pull llama3.1              # chat/reasoning model -> powers the agent + tools

Install the one Python dependency:
    pip install requests

Run:
    python hr_agent_ollama_rag.py
"""

import json
import math
import requests

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.1"


# =============================================================================
# 1. RAG — Retrieval-Augmented Generation layer
# =============================================================================
# Every resume is chunked, embedded, and stored. When the agent evaluates a
# candidate against a job description, it retrieves only the resume chunks
# most relevant to the JD instead of dumping the whole resume into the
# prompt — this is the "retrieval" half of RAG, and it's what keeps the
# model's output grounded in the candidate's real, uploaded document.

def chunk_text(text, max_words=60):
    """Split resume/JD text into small overlapping chunks for embedding."""
    words = text.split()
    chunks = []
    step = max_words - 10  # 10-word overlap keeps context from splitting mid-idea
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + max_words])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def embed(text):
    """Call Ollama's embedding endpoint and return a vector."""
    resp = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b + 1e-8)


class VectorStore:
    """A minimal in-memory vector store — one per candidate resume."""

    def __init__(self):
        self.records = []  # [{text, embedding, source}]

    def add(self, text, source):
        for chunk in chunk_text(text):
            self.records.append({"text": chunk, "embedding": embed(chunk), "source": source})

    def retrieve(self, query, top_k=4):
        """RAG retrieval step: rank stored chunks by similarity to the query."""
        query_vec = embed(query)
        scored = [
            (cosine_similarity(query_vec, r["embedding"]), r) for r in self.records
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r["text"] for _, r in scored[:top_k]]


# =============================================================================
# 2. TOOLS — structured functions the agent can call
# =============================================================================
# Ollama's chat models don't have Claude-style native tool-use, so structured
# tool output is enforced with `format="json"` plus a strict schema in the
# prompt. Each function below is one callable "tool" available to the agent.

def call_ollama_json(prompt, system):
    """Low-level helper: call the local chat model and force JSON output."""
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": CHAT_MODEL,
            "format": "json",
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=120,
    )
    resp.raise_for_status()
    content = resp.json()["message"]["content"]
    return json.loads(content)


def tool_evaluate_candidate(jd_text, retrieved_resume_chunks, candidate_name):
    """Tool 1: score a candidate against the job description using retrieved context."""
    context = "\n---\n".join(retrieved_resume_chunks)
    schema = (
        "Return ONLY valid JSON with exactly these keys: "
        "candidate_name (string), match_score (integer 0-100), "
        "years_experience (number), matched_skills (array of strings), "
        "missing_skills (array of strings), summary (string, 1-2 sentences), "
        "verdict (one of: 'Strong Match', 'Possible Fit', 'Not a Fit')."
    )
    prompt = (
        f"JOB DESCRIPTION:\n{jd_text}\n\n"
        f"MOST RELEVANT RESUME EXCERPTS (retrieved via RAG):\n{context}\n\n"
        f"Candidate name hint: {candidate_name}\n\n{schema}"
    )
    system = "You are a meticulous HR recruitment agent. Be honest about gaps, not generous."
    return call_ollama_json(prompt, system)


def tool_generate_interview_questions(jd_text, retrieved_resume_chunks, evaluation):
    """Tool 2: generate interview questions tailored to this candidate's gaps."""
    context = "\n---\n".join(retrieved_resume_chunks)
    schema = (
        "Return ONLY valid JSON: {\"questions\": [{\"category\": string, "
        "\"question\": string}, ...]} with exactly 5 items — include at least "
        "one probing a missing skill, one technical, one behavioral."
    )
    prompt = (
        f"JOB DESCRIPTION:\n{jd_text}\n\n"
        f"RELEVANT RESUME EXCERPTS:\n{context}\n\n"
        f"MATCHED SKILLS: {evaluation.get('matched_skills')}\n"
        f"MISSING SKILLS: {evaluation.get('missing_skills')}\n\n{schema}"
    )
    system = "You are an HR agent preparing an interviewer with sharp, specific questions."
    return call_ollama_json(prompt, system)


def tool_rank_candidates(evaluations):
    """Tool 3: deterministic ranking tool (no LLM call needed — plain logic)."""
    return sorted(evaluations, key=lambda e: e["match_score"], reverse=True)


# =============================================================================
# 3. AGENT — orchestrates RAG + tools into a multi-step workflow
# =============================================================================
#
# Agent name:     HR Recruitment Screening Agent
# Role:           Screens candidate resumes against a job description and
#                 prepares interviewers with tailored, grounded questions.
# Inputs:         one job description (text) + one or more resumes (text)
# Memory:         a VectorStore per run, holding embedded resume chunks
# Tools it calls: tool_evaluate_candidate, tool_generate_interview_questions,
#                 tool_rank_candidates
# Loop:           for each candidate -> RETRIEVE (RAG) -> ACT (evaluate tool)
#                 -> after all candidates -> RANK -> on demand -> ACT (questions tool)
# Output:         a ranked, structured screening report per candidate

class HRRecruitmentAgent:
    def __init__(self, job_description):
        self.jd = job_description
        self.store = VectorStore()
        self.candidates = {}  # name -> raw resume text

    def add_candidate(self, name, resume_text):
        self.candidates[name] = resume_text
        self.store.add(resume_text, source=name)

    def screen_all(self):
        """Step 1: RAG-retrieve + evaluate every candidate against the JD."""
        evaluations = []
        for name, resume_text in self.candidates.items():
            retrieved = self.store.retrieve(self.jd, top_k=4)
            eval_result = tool_evaluate_candidate(self.jd, retrieved, name)
            evaluations.append(eval_result)
        return tool_rank_candidates(evaluations)

    def prepare_interview(self, name, evaluation):
        """Step 2 (on demand): RAG-retrieve again + generate tailored questions."""
        retrieved = self.store.retrieve(self.jd, top_k=4)
        return tool_generate_interview_questions(self.jd, retrieved, evaluation)


# =============================================================================
# Example run
# =============================================================================
if __name__ == "__main__":
    jd = """Backend Developer — 2+ years experience with Java, Spring Boot,
    REST APIs and MySQL. Familiarity with Docker and CI/CD is a plus."""

    agent = HRRecruitmentAgent(jd)
    agent.add_candidate("Arjun", "Java developer, 3 years, Spring Boot, MySQL, no Docker experience.")
    agent.add_candidate("Divya", "Frontend React developer, 1 year, no backend experience.")

    ranked = agent.screen_all()
    print(json.dumps(ranked, indent=2))

    top_candidate = ranked[0]
    questions = agent.prepare_interview(top_candidate["candidate_name"], top_candidate)
    print(json.dumps(questions, indent=2))
