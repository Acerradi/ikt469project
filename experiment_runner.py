#!/usr/bin/env python3
"""
experiment_runner.py

Systematically evaluates RAG pipeline configurations for the IKT469 UiA
course catalogue chatbot. Experiments vary across:
  - Embedding model        (affects retrieval quality)
  - Chunk size / overlap   (affects retrieval precision vs. context richness)
  - k (retrieved docs)     (affects how much context the generator sees)
  - Generator model        (affects answer quality)
  - Judge model            (affects evaluation scores)

Results are appended to a CSV after each complete experiment so partial
progress is preserved if the script is interrupted.

Usage:
    python experiment_runner.py
    python experiment_runner.py --output my_results.csv
    python experiment_runner.py --skip-index-rebuild   # use existing chroma_langchain_db
"""

import argparse
import csv
import json
import os
import shutil
import time
import datetime
from pathlib import Path
from typing import Optional

import psutil
import re

import pandas as pd
import requests
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

# ─── Paths ────────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434"
CSV_DATA_PATH = "./uia_ikt_courses.csv"
DEFAULT_OUTPUT = "./experiment_results.csv"
BASELINE_CHROMA_DIR = "./chroma_langchain_db"
TEMP_INDEX_BASE = "./chroma_experiment_tmp"

# ─── Fields matching collect_store_data.ipynb ─────────────────────────────────

SCALAR_FIELDS = [
    "title", "url", "ects_credits", "responsible_department",
    "course_leader", "course_leaders", "lecture_semester",
    "teaching_language", "duration",
]

SECTION_FIELDS = [
    "learning_outcomes", "contents", "teaching_and_learning_methods",
    "examinations", "recommended_previous_knowledge", "examination_requirements",
    "prerequisites", "other_information", "student_evaluation",
    "reduction_of_credits",
    "the_course_is_connected_to_the_following_study_programs",
    "offered_as_a_free_standing_course",
    "admission_requirement_if_given_as_a_free_standing_course",
]

# ─── Experiment grid ──────────────────────────────────────────────────────────
#
# Each entry: (label, embedding_model, chunk_size, chunk_overlap, is_baseline)
#
# is_baseline=True means this config matches the existing chroma_langchain_db,
# so we can fall back to it when the source CSV is unavailable.

EMBED_CHUNK_CONFIGS = [
    (
        "minilm-500",
        "sentence-transformers/all-MiniLM-L6-v2",
        500, 100,
        True,   # matches existing ./chroma_langchain_db
    ),
    (
        "minilm-250",
        "sentence-transformers/all-MiniLM-L6-v2",
        250, 50,
        False,
    ),
    (
        "mpnet-500",
        "sentence-transformers/all-mpnet-base-v2",
        500, 100,
        False,
    ),
]

K_VALUES = [3, 5, 10]

# Filtered at runtime to models that are actually pulled in Ollama
CANDIDATE_GENERATOR_MODELS = ["llama3.2:3b", "qwen2.5:0.5b"]
CANDIDATE_JUDGE_MODELS = ["llama3.2:3b"]

COURSE_CODE_RE = re.compile(r"\bIKT\d{3}\b")

RAG_PROMPT = """\
You are a course information assistant for UiA (University of Agder).
Answer using ONLY the information provided in the context below. Do not use prior knowledge about any courses.

Rules:
- If the information is not in the context, say exactly: "The information is not provided in the retrieved context."
- Factual questions (ECTS credits, instructor, language, semester): give a brief, direct answer.
- Comparison questions: address each course mentioned in the question separately and systematically.
- List or synthesis questions: enumerate every matching item found in the context; do not assume there are more beyond what the context shows.
- Never infer or guess anything not explicitly stated.

Context:
{context}

Question: {question}

Answer:"""

# ─── Judge prompts (identical to evaluate.py) ─────────────────────────────────

GROUNDING_PROMPT = """
You are evaluating a RAG system.

Question:
{question}

Retrieved Context:
{context}

Answer:
{answer}

Task:
Is the answer fully supported by the context?

Return ONLY valid JSON:
{{
  "score": float (0 to 1),
  "verdict": "pass" or "fail",
  "unsupported_claims": [list],
  "reason": "short explanation"
}}

Rules:
- 1.0 = fully grounded in context
- 0.0 = hallucinated or unsupported
- Use ONLY the context
- Be strict
"""

RELEVANCE_PROMPT = """
You are evaluating answer quality.

Question:
{question}

Answer:
{answer}

Task:
Does the answer properly answer the question?

Return ONLY valid JSON:
{{
  "score": float (0 to 1),
  "verdict": "pass" or "fail",
  "missing_points": [list],
  "reason": "short explanation"
}}

Rules:
- Penalize vague or incomplete answers
- Penalize irrelevant info
"""

# ─── Question set (from evaluate.py) ──────────────────────────────────────────

QUESTION_SET = [
    {
        "id": "q001",
        "question": "Which course is taught by Morten Goodwin?",
        "expected_answer": "IKT469 Deep Neural Networks (Spring 2026).",
        "answer_type": "factoid",
        "difficulty": "easy",
        "course_scope": ["IKT469"],
        "evidence_fields": ["course_leader", "title"],
    },
    {
        "id": "q002",
        "question": "How many ECTS credits is IKT469 Deep Neural Networks worth?",
        "expected_answer": "7.5 ECTS.",
        "answer_type": "factoid",
        "difficulty": "easy",
        "course_scope": ["IKT469"],
        "evidence_fields": ["title", "ects_credits"],
    },
    {
        "id": "q003",
        "question": "What is the teaching language of IKT112 Concepts of Machine Learning?",
        "expected_answer": "English.",
        "answer_type": "factoid",
        "difficulty": "easy",
        "course_scope": ["IKT112"],
        "evidence_fields": ["title", "teaching_language"],
    },
    {
        "id": "q004",
        "question": "Who is the course leader for IKT204 Data communication?",
        "expected_answer": "Sigurd Eskeland.",
        "answer_type": "factoid",
        "difficulty": "easy",
        "course_scope": ["IKT204"],
        "evidence_fields": ["title", "course_leader"],
    },
    {
        "id": "q005",
        "question": "In which semester is IKT443 WiFi and Internet of Things offered?",
        "expected_answer": "Spring.",
        "answer_type": "factoid",
        "difficulty": "easy",
        "course_scope": ["IKT443"],
        "evidence_fields": ["title", "lecture_semester"],
    },
    {
        "id": "q006",
        "question": "Is IKT115 Introduction to Artificial Intelligence Technology offered as a free-standing course?",
        "expected_answer": "No. It is available as a continuing education course (IKT902), but not as a free-standing course.",
        "answer_type": "factoid",
        "difficulty": "easy",
        "course_scope": ["IKT115"],
        "evidence_fields": ["title", "offered_as_a_free_standing_course"],
    },
    {
        "id": "q007",
        "question": "What is the duration of IKT302 Bachelor's Thesis, data?",
        "expected_answer": "½ year.",
        "answer_type": "factoid",
        "difficulty": "easy",
        "course_scope": ["IKT302"],
        "evidence_fields": ["title", "duration"],
    },
    {
        "id": "q008",
        "question": "Which course is led by Andreas Prinz?",
        "expected_answer": "IKT122 Success – AI Tools and Mindset for Goal Achievement (Spring 2026).",
        "answer_type": "factoid",
        "difficulty": "easy",
        "course_scope": ["IKT122"],
        "evidence_fields": ["course_leader", "title"],
    },
    {
        "id": "q009",
        "question": "What are the examination components for IKT112 Concepts of Machine Learning?",
        "expected_answer": "A 3-hour written exam worth 50% and a portfolio assessment worth 50%, with graded assessment.",
        "answer_type": "descriptive",
        "difficulty": "medium",
        "course_scope": ["IKT112"],
        "evidence_fields": ["title", "examinations"],
    },
    {
        "id": "q010",
        "question": "Summarize the main learning outcomes of IKT469 Deep Neural Networks.",
        "expected_answer": "Students should understand advanced deep learning concepts, transformer models, multimodal representation learning, generative models, explainability methods, advanced optimization, graph and sequence models, embeddings, retrieval-augmented systems, and be able to design, implement, evaluate, and critically assess advanced neural architectures.",
        "answer_type": "summary",
        "difficulty": "medium",
        "course_scope": ["IKT469"],
        "evidence_fields": ["title", "learning_outcomes"],
    },
    {
        "id": "q011",
        "question": "What teaching and learning methods are used in IKT460 Reinforcement Learning?",
        "expected_answer": "Combination of lectures, assignments, paper studies, lab work, report writing, and self-study. Tasks are done individually or in small groups of 2 students with group supervision.",
        "answer_type": "descriptive",
        "difficulty": "medium",
        "course_scope": ["IKT460"],
        "evidence_fields": ["title", "teaching_and_learning_methods"],
    },
    {
        "id": "q012",
        "question": "What are the admission requirements and recommended previous knowledge for IKT204 Data communication?",
        "expected_answer": "Admission requires Higher Education Entrance Qualification including mathematics R1 and R2 and physics Fysikk 1, or a pass in the preliminary course examination for engineers. Recommended previous knowledge is basic networking and security understanding such as IKT100-G Network, security, and privacy.",
        "answer_type": "descriptive",
        "difficulty": "medium",
        "course_scope": ["IKT204"],
        "evidence_fields": [
            "title",
            "admission_requirement_if_given_as_a_free_standing_course",
            "recommended_previous_knowledge",
        ],
    },
    {
        "id": "q013",
        "question": "What are the prerequisites for starting IKT302 Bachelor's Thesis, data?",
        "expected_answer": "The student must have passed at least 130 ECTS credits in their study plan by the start of the semester, and the topic and research question must be approved by the course leader for the bachelor thesis.",
        "answer_type": "descriptive",
        "difficulty": "medium",
        "course_scope": ["IKT302"],
        "evidence_fields": ["title", "prerequisites"],
    },
    {
        "id": "q014",
        "question": "What are the compulsory requirements for IKT590 Master's Thesis?",
        "expected_answer": "There must be 10 compulsory guidance meetings for every student/group, and the thesis press release must be submitted and approved.",
        "answer_type": "descriptive",
        "difficulty": "medium",
        "course_scope": ["IKT590"],
        "evidence_fields": ["title", "examination_requirements"],
    },
    {
        "id": "q015",
        "question": "Compare the assessment methods of IKT469 Deep Neural Networks and IKT468 Applied Algorithms.",
        "expected_answer": "IKT469 uses graded portfolio assessment only, with a postponed exam available for the portfolio. IKT468 uses a portfolio with project assignments worth 60% and a 3-hour written examination worth 40%.",
        "answer_type": "comparison",
        "difficulty": "medium",
        "course_scope": ["IKT469", "IKT468"],
        "evidence_fields": ["title", "examinations"],
    },
    {
        "id": "q016",
        "question": "Which has more ECTS credits: IKT112 Concepts of Machine Learning or IKT215 Pattern Recognition?",
        "expected_answer": "IKT215 Pattern Recognition has more credits. IKT112 is 5 ECTS, while IKT215 is 10 ECTS.",
        "answer_type": "comparison",
        "difficulty": "easy",
        "course_scope": ["IKT112", "IKT215"],
        "evidence_fields": ["title", "ects_credits"],
    },
    {
        "id": "q017",
        "question": "Compare the workload of IKT112, IKT215, and IKT469.",
        "expected_answer": "IKT112 has approximately 135 hours of workload, while IKT215 and IKT469 each have approximately 200 hours.",
        "answer_type": "comparison",
        "difficulty": "medium",
        "course_scope": ["IKT112", "IKT215", "IKT469"],
        "evidence_fields": ["title", "teaching_and_learning_methods"],
    },
    {
        "id": "q018",
        "question": "Which course has a larger exam component: IKT112 Concepts of Machine Learning or IKT443 WiFi and Internet of Things?",
        "expected_answer": "IKT443 has the larger exam component. Its individual 4-hour written exam counts 75% of the final grade, while IKT112 has a 3-hour written exam worth 50%.",
        "answer_type": "comparison",
        "difficulty": "medium",
        "course_scope": ["IKT112", "IKT443"],
        "evidence_fields": ["title", "examinations"],
    },
    {
        "id": "q019",
        "question": "Compare IKT452 Computer Vision and IKT460 Reinforcement Learning in terms of course content focus.",
        "expected_answer": "IKT452 focuses on computer vision topics such as image formation, camera geometry, feature detection, matching, stereo, motion estimation, tracking, scene understanding, and deep neural networks for vision. IKT460 focuses on reinforcement learning topics such as Markov decision processes, bandits, policy iteration, dynamic programming, TD-learning, direct policy search, deep RL, and end-to-end RL pipelines.",
        "answer_type": "comparison",
        "difficulty": "medium",
        "course_scope": ["IKT452", "IKT460"],
        "evidence_fields": ["title", "contents"],
    },
    {
        "id": "q020",
        "question": "Which course is most directly focused on deep learning and retrieval-augmented systems?",
        "expected_answer": "IKT469 Deep Neural Networks, because it explicitly includes advanced deep learning, transformer models, multimodal learning, embeddings, and retrieval-augmented generation with external knowledge integration.",
        "answer_type": "recommendation",
        "difficulty": "medium",
        "course_scope": ["IKT469"],
        "evidence_fields": ["title", "learning_outcomes", "contents"],
    },
    {
        "id": "q021",
        "question": "Which course would best fit a student who wants to learn reinforcement learning specifically?",
        "expected_answer": "IKT460 Reinforcement Learning.",
        "answer_type": "recommendation",
        "difficulty": "easy",
        "course_scope": ["IKT460"],
        "evidence_fields": ["title", "contents", "learning_outcomes"],
    },
    {
        "id": "q022",
        "question": "Which course is the best match for learning computer vision with hands-on implementation?",
        "expected_answer": "IKT452 Computer Vision, because it covers both theoretical and practical aspects of computing with images and includes hands-on experience solving real-life vision problems.",
        "answer_type": "recommendation",
        "difficulty": "medium",
        "course_scope": ["IKT452"],
        "evidence_fields": ["title", "contents", "learning_outcomes"],
    },
    {
        "id": "q023",
        "question": "Which courses appear most relevant for a student interested in IoT networking and communication technologies?",
        "expected_answer": "IKT443 WiFi and Internet of Things, IKT458 5G and IoT: Advanced, and IKT520 Security in IoT and Machine-Type Communication. IKT204 Data communication is also relevant as a foundational networking course.",
        "answer_type": "synthesis",
        "difficulty": "hard",
        "course_scope": ["IKT443", "IKT458", "IKT520", "IKT204"],
        "evidence_fields": ["title", "contents", "learning_outcomes"],
    },
    {
        "id": "q024",
        "question": "If a student wants a course on intelligent data processing and large-scale data-driven decision making, which course fits best?",
        "expected_answer": "IKT453 Intelligent Data Management.",
        "answer_type": "recommendation",
        "difficulty": "easy",
        "course_scope": ["IKT453"],
        "evidence_fields": ["title", "contents", "learning_outcomes"],
    },
    {
        "id": "q025",
        "question": "Can a student take IKT204 Data communication as a free-standing course?",
        "expected_answer": "Yes, subject to availability or capacity.",
        "answer_type": "factoid",
        "difficulty": "easy",
        "course_scope": ["IKT204"],
        "evidence_fields": ["title", "offered_as_a_free_standing_course"],
    },
    {
        "id": "q026",
        "question": "What must students complete before they can take the exam in IKT443 WiFi and Internet of Things?",
        "expected_answer": "Compulsory assignments must be approved.",
        "answer_type": "factoid",
        "difficulty": "easy",
        "course_scope": ["IKT443"],
        "evidence_fields": ["title", "examination_requirements"],
    },
    {
        "id": "q027",
        "question": "What prior knowledge is recommended for IKT452 Computer Vision?",
        "expected_answer": "IKT213 - Machine Vision.",
        "answer_type": "factoid",
        "difficulty": "easy",
        "course_scope": ["IKT452"],
        "evidence_fields": ["title", "recommended_previous_knowledge"],
    },
    {
        "id": "q028",
        "question": "What prior knowledge is recommended for IKT519 Blockchain and distributed ledger technology?",
        "expected_answer": "It is recommended that students have prior knowledge corresponding to IKT447-G Trust, threats, risks and vulnerabilities.",
        "answer_type": "factoid",
        "difficulty": "easy",
        "course_scope": ["IKT519"],
        "evidence_fields": [
            "title",
            "admission_requirement_if_given_as_a_free_standing_course",
            "recommended_previous_knowledge",
        ],
    },
    {
        "id": "q029",
        "question": "What are the minimum requirements to start IKT590 Master's Thesis?",
        "expected_answer": "Students must have passed examinations with no less than 75 ECTS credits in the master programme at the beginning of the semester, and the thesis topic must be approved by the University of Agder.",
        "answer_type": "descriptive",
        "difficulty": "medium",
        "course_scope": ["IKT590"],
        "evidence_fields": ["title", "prerequisites"],
    },
    {
        "id": "q030",
        "question": "Who is the course leader for IKT433 Distributed and Big Data Systems?",
        "expected_answer": "The provided data does not specify a course leader for IKT433.",
        "answer_type": "unanswerable",
        "difficulty": "easy",
        "course_scope": ["IKT433"],
        "evidence_fields": ["title", "course_leader"],
        "is_unanswerable": True,
    },
    {
        "id": "q031",
        "question": "What is the exact duration of IKT433 Distributed and Big Data Systems?",
        "expected_answer": "The provided data does not clearly specify the duration for IKT433.",
        "answer_type": "unanswerable",
        "difficulty": "easy",
        "course_scope": ["IKT433"],
        "evidence_fields": ["title", "duration"],
        "is_unanswerable": True,
    },
    {
        "id": "q032",
        "question": "Which textbook is required for IKT469 Deep Neural Networks?",
        "expected_answer": "The provided data does not list any required textbook for IKT469.",
        "answer_type": "unanswerable",
        "difficulty": "easy",
        "course_scope": ["IKT469"],
        "evidence_fields": ["title"],
        "is_unanswerable": True,
    },
    {
        "id": "q033",
        "question": "On which weekday are lectures for IKT460 held?",
        "expected_answer": "The provided data does not include lecture weekdays or schedule details for IKT460.",
        "answer_type": "unanswerable",
        "difficulty": "easy",
        "course_scope": ["IKT460"],
        "evidence_fields": ["title"],
        "is_unanswerable": True,
    },
    {
        "id": "q034",
        "question": "What is the classroom number for IKT122?",
        "expected_answer": "The provided data does not include classroom information for IKT122.",
        "answer_type": "unanswerable",
        "difficulty": "easy",
        "course_scope": ["IKT122"],
        "evidence_fields": ["title"],
        "is_unanswerable": True,
    },
    {
        "id": "q035",
        "question": "Which spring 2026 courses in the dataset explicitly mention machine learning in either the title, learning outcomes, or contents?",
        "expected_answer": "At minimum: IKT112 Concepts of Machine Learning, IKT115 Introduction to Artificial Intelligence Technology, IKT215 Pattern Recognition, IKT459 Embedded Sensors, Signal Processing and Machine Learning for Autonomous Systems, IKT469 Deep Neural Networks. IKT460 Reinforcement Learning is also clearly machine learning related even though its title is more specific.",
        "answer_type": "synthesis",
        "difficulty": "hard",
        "course_scope": ["IKT112", "IKT115", "IKT215", "IKT459", "IKT469", "IKT460"],
        "evidence_fields": ["title", "learning_outcomes", "contents"],
    },
    {
        "id": "q036",
        "question": "Which courses in the dataset are 30 ECTS thesis courses?",
        "expected_answer": "IKT302 Bachelor's Thesis, data; IKT523 Master's Thesis, Cyber Security; and IKT590 Master's Thesis.",
        "answer_type": "list",
        "difficulty": "easy",
        "course_scope": ["IKT302", "IKT523", "IKT590"],
        "evidence_fields": ["title", "ects_credits"],
    },
    {
        "id": "q037",
        "question": "Which courses are taught in English and are available as free-standing courses, subject to availability?",
        "expected_answer": "Examples include IKT112, IKT215, IKT433, IKT443, IKT449, IKT452, IKT453, IKT458, IKT459, IKT460, IKT462, and IKT468, based on the provided records stating English teaching language and 'Yes' or equivalent wording for free-standing availability.",
        "answer_type": "synthesis",
        "difficulty": "hard",
        "course_scope": ["multiple"],
        "evidence_fields": ["title", "teaching_language", "offered_as_a_free_standing_course"],
    },
    {
        "id": "q038",
        "question": "Which courses explicitly require compulsory assignments or exercises to be approved before examination?",
        "expected_answer": "At least IKT204, IKT433, IKT443, IKT449, IKT519, and IKT520 explicitly state approval of compulsory exercises, assignments, hand-ins, or presentations as a requirement before the exam or examination.",
        "answer_type": "synthesis",
        "difficulty": "hard",
        "course_scope": ["IKT204", "IKT433", "IKT443", "IKT449", "IKT519", "IKT520"],
        "evidence_fields": ["title", "examination_requirements"],
    },
    {
        "id": "q039",
        "question": "Which courses appear most relevant for a student interested in cybersecurity?",
        "expected_answer": "IKT449 Selected Security Topics, IKT519 Blockchain and distributed ledger technology, IKT520 Security in IoT and Machine-Type Communication, and IKT523 Master's Thesis, Cyber Security. IKT204 also includes operational security and basic cryptography as supporting knowledge.",
        "answer_type": "synthesis",
        "difficulty": "hard",
        "course_scope": ["IKT449", "IKT519", "IKT520", "IKT523", "IKT204"],
        "evidence_fields": ["title", "contents", "learning_outcomes"],
    },
    {
        "id": "q040",
        "question": "Which course best matches a student wanting to build multimodal and retrieval-augmented systems using embeddings and external knowledge?",
        "expected_answer": "IKT469 Deep Neural Networks.",
        "answer_type": "recommendation",
        "difficulty": "medium",
        "course_scope": ["IKT469"],
        "evidence_fields": ["title", "learning_outcomes", "contents"],
    },
]

# ─── Helpers ──────────────────────────────────────────────────────────────────

def check_ollama() -> bool:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def get_available_models() -> list:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        data = r.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def safe_parse_json(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])
    except Exception:
        pass
    return None


def normalize_score(value) -> float:
    try:
        v = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, v))


def clean_text(text) -> str:
    if text is None:
        return ""
    return str(text).replace("\r", " ").replace("\n", " \\n ").strip()


RAPL_ENERGY_PATH = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"
RAPL_MAX_PATH = "/sys/class/powercap/intel-rapl/intel-rapl:0/max_energy_range_uj"


def _read_rapl_uj() -> Optional[float]:
    try:
        return float(Path(RAPL_ENERGY_PATH).read_text().strip())
    except Exception:
        return None


def _energy_delta_j(start_uj: Optional[float], end_uj: Optional[float]) -> Optional[float]:
    if start_uj is None or end_uj is None:
        return None
    delta = end_uj - start_uj
    if delta < 0:  # counter wrapped around its max value
        try:
            delta += float(Path(RAPL_MAX_PATH).read_text().strip())
        except Exception:
            return None
    return round(delta / 1_000_000, 4)  # µJ → J


def _process_cpu_time_s() -> float:
    t = psutil.Process().cpu_times()
    return t.user + t.system


def run_judge(judge_llm: ChatOllama, prompt_template: str, question: str, context: str, answer: str) -> dict:
    prompt = prompt_template.format(question=question, context=context, answer=answer)
    response = judge_llm.invoke(prompt)
    parsed = safe_parse_json(response.content)
    if parsed is not None:
        parsed["score"] = normalize_score(parsed.get("score", 0.0))
        return parsed
    return {"score": 0.0, "verdict": "fail", "reason": "Invalid JSON from judge"}


def build_documents(df: pd.DataFrame, splitter: RecursiveCharacterTextSplitter) -> list:
    docs = []
    for _, row in df.iterrows():
        meta = {f: str(row.get(f, "")) for f in SCALAR_FIELDS}
        meta["row_id"] = str(row.name)

        summary_lines = [
            f"{f.replace('_', ' ').title()}: {row[f]}"
            for f in SCALAR_FIELDS
            if pd.notna(row.get(f)) and str(row.get(f, "")).strip()
        ]
        docs.append(Document(
            page_content="\n".join(summary_lines),
            metadata={**meta, "section": "summary", "chunk_id": 0},
        ))

        for field in SECTION_FIELDS:
            val = row.get(field)
            if pd.isna(val) or not str(val).strip():
                continue
            header = f"{field.replace('_', ' ').title()} ({meta.get('title', '')})"
            chunks = splitter.split_text(f"{header}\n{val}")
            for i, chunk in enumerate(chunks):
                docs.append(Document(
                    page_content=chunk,
                    metadata={**meta, "section": field, "chunk_id": i},
                ))
    return docs


def build_catalog_documents(df: pd.DataFrame, splitter: RecursiveCharacterTextSplitter) -> list:
    """One compact line per course → split into chunks.

    Synthesis queries ("which courses mention X?", "which are 30 ECTS?") need to
    scan across all courses. A full-corpus catalog document makes that possible
    without increasing k to cover every individual section chunk.
    """
    lines = ["UiA IKT Spring 2026 — Full Course Catalog\n"]
    for _, row in df.iterrows():
        url = str(row.get("url", ""))
        m = re.search(r"ikt(\d{3})", url, re.IGNORECASE)
        code = f"IKT{m.group(1).upper()}" if m else ""

        title = str(row.get("title", "")).strip()
        ects = str(row.get("ects_credits", "")).strip()
        lang = str(row.get("teaching_language", "")).strip()
        free = str(row.get("offered_as_a_free_standing_course", "")).strip()
        leader = str(row.get("course_leader", "")).strip()
        exam_req = str(row.get("examination_requirements", "")) if pd.notna(row.get("examination_requirements")) else ""
        exam_snippet = exam_req[:80].replace("\n", " ").strip()

        topics_raw = " ".join(filter(None, [
            str(row.get("learning_outcomes", "")) if pd.notna(row.get("learning_outcomes")) else "",
            str(row.get("contents", "")) if pd.notna(row.get("contents")) else "",
        ]))
        topics_snippet = topics_raw[:120].replace("\n", " ").strip()

        lines.append(
            f"{code} {title}: {ects} ECTS | {lang} | free-standing: {free} | "
            f"leader: {leader} | exam requirements: {exam_snippet} | topics: {topics_snippet}"
        )

    chunks = splitter.split_text("\n".join(lines))
    return [
        Document(page_content=chunk, metadata={"section": "catalog_overview", "chunk_id": i, "row_id": "catalog"})
        for i, chunk in enumerate(chunks)
    ]


def augmented_retrieve(vectorstore: Chroma, retriever, question: str) -> list:
    """Semantic retrieval + course-code injection.

    For questions that name specific IKT courses (e.g. "Compare IKT452 and IKT460"),
    semantic top-k alone may miss one of the courses entirely. This function adds up
    to 3 extra chunks per named course code to guarantee coverage.
    """
    docs = retriever.invoke(question)
    seen = {d.page_content[:80] for d in docs}

    for code in set(COURSE_CODE_RE.findall(question.upper())):
        for doc in vectorstore.similarity_search(code, k=3):
            key = doc.page_content[:80]
            if key not in seen:
                docs.append(doc)
                seen.add(key)

    return docs


def build_vectorstore(
    label: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    persist_dir: str,
) -> Chroma:
    print(f"\n  Building index [{label}]: {embedding_model}, chunk={chunk_size}/{chunk_overlap}")
    df = pd.read_csv(CSV_DATA_PATH)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
    os.makedirs(persist_dir, exist_ok=True)

    docs = build_documents(df, splitter)
    catalog_docs = build_catalog_documents(df, splitter)
    docs = docs + catalog_docs
    print(f"  → {len(docs)} documents from {len(df)} courses ({len(catalog_docs)} catalog chunks)")

    vs = Chroma(
        persist_directory=persist_dir,
        collection_name="uia_courses",
        embedding_function=embeddings,
    )
    batch_size = 32
    for i in tqdm(range(0, len(docs), batch_size), desc=f"  Indexing [{label}]", leave=False):
        vs.add_documents(docs[i : i + batch_size])

    return vs


def load_baseline_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        persist_directory=BASELINE_CHROMA_DIR,
        collection_name="uia_courses",
        embedding_function=embeddings,
    )


def append_to_csv(rows: list, output_path: str) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    write_header = not Path(output_path).exists()
    df.to_csv(
        output_path,
        mode="a",
        index=False,
        header=write_header,
        quoting=csv.QUOTE_ALL,
    )


def load_completed_experiments(summary_path: str) -> set:
    """Return config tuples for every experiment that has a completed summary row."""
    if not Path(summary_path).exists():
        return set()
    try:
        df = pd.read_csv(summary_path)
    except Exception:
        return set()
    completed = set()
    for _, row in df.iterrows():
        completed.add((
            str(row["embedding_model"]),
            int(row["chunk_size"]),
            int(row["chunk_overlap"]),
            int(row["k"]),
            str(row["generator_model"]),
            str(row["judge_model"]),
        ))
    return completed


def remove_partial_rows(output_path: str, completed: set) -> None:
    """Strip question rows whose experiment never wrote a summary row.

    These are left over from a run that was interrupted mid-experiment.
    Without this cleanup, resuming would append duplicate rows for the
    questions that completed before the interruption.
    """
    if not Path(output_path).exists():
        return
    try:
        df = pd.read_csv(output_path)
    except Exception:
        return
    if df.empty:
        return

    required = {"embedding_model", "chunk_size", "chunk_overlap", "k", "generator_model", "judge_model"}
    if not required.issubset(df.columns):
        return

    def _belongs_to_completed(row):
        return (
            str(row["embedding_model"]),
            int(row["chunk_size"]),
            int(row["chunk_overlap"]),
            int(row["k"]),
            str(row["generator_model"]),
            str(row["judge_model"]),
        ) in completed

    mask = df.apply(_belongs_to_completed, axis=1)
    n_partial = int((~mask).sum())
    if n_partial:
        print(f"  Removing {n_partial} partial rows from {output_path}")
        df[mask].to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="RAG experiment runner for IKT469")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument(
        "--skip-index-rebuild",
        action="store_true",
        help="Use the existing chroma_langchain_db for the baseline config; skip non-baseline configs that require CSV",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip experiments already present in the summary CSV; strip partial rows from any interrupted experiment",
    )
    args = parser.parse_args()

    # ── Preflight checks ──────────────────────────────────────────────────────
    if not check_ollama():
        print("ERROR: Ollama is not reachable at http://localhost:11434")
        print("       Start it with:  docker compose up -d")
        return

    available_models = get_available_models()
    print(f"Ollama models available: {available_models}")

    generator_models = [m for m in CANDIDATE_GENERATOR_MODELS if m in available_models]
    judge_models = [m for m in CANDIDATE_JUDGE_MODELS if m in available_models]

    if not generator_models:
        print(f"ERROR: None of the generator models are pulled in Ollama.")
        print(f"       Wanted: {CANDIDATE_GENERATOR_MODELS}")
        print(f"       Pull with: docker exec -it <container> ollama pull llama3.2:3b")
        return
    if not judge_models:
        print(f"ERROR: None of the judge models are pulled in Ollama.")
        print(f"       Wanted: {CANDIDATE_JUDGE_MODELS}")
        return

    csv_available = Path(CSV_DATA_PATH).exists()
    if not csv_available:
        print(f"WARNING: {CSV_DATA_PATH} not found.")
        print("         Only the baseline index (all-MiniLM-L6-v2, chunk=500/100) will be used.")
        print("         Regenerate it with collect_store_data.ipynb to unlock all configs.\n")

    # ── Determine which embed/chunk configs to run ────────────────────────────
    if args.skip_index_rebuild or not csv_available:
        active_configs = [c for c in EMBED_CHUNK_CONFIGS if c[4]]  # is_baseline only
    else:
        active_configs = EMBED_CHUNK_CONFIGS

    total_experiments = len(active_configs) * len(K_VALUES) * len(generator_models) * len(judge_models)
    total_questions = total_experiments * len(QUESTION_SET)

    summary_path = str(Path(args.output).with_suffix("")) + "_summary.csv"

    # ── Resume logic ──────────────────────────────────────────────────────────
    completed: set = set()
    if args.resume:
        completed = load_completed_experiments(summary_path)
        if completed:
            print(f"Resuming: {len(completed)} experiment(s) already complete, skipping them.")
            remove_partial_rows(args.output, completed)
        else:
            print("--resume: no completed experiments found in summary CSV, starting fresh.")
        print()

    print(f"Generator models : {generator_models}")
    print(f"Judge models     : {judge_models}")
    print(f"Embed/chunk configs: {[c[0] for c in active_configs]}")
    print(f"k values         : {K_VALUES}")
    print(f"Experiments      : {total_experiments}")
    print(f"Questions total  : {total_questions}")
    print(f"Output CSV       : {args.output}")
    print(f"Summary CSV      : {summary_path}\n")

    # ── Run experiments ───────────────────────────────────────────────────────
    exp_counter = 0

    with tqdm(total=total_experiments, desc="Overall progress", unit="exp", position=0) as outer:
        for label, embedding_model, chunk_size, chunk_overlap, is_baseline in active_configs:

            # Skip vectorstore build entirely when every experiment in this group is done
            group_keys = [
                (embedding_model, chunk_size, chunk_overlap, k, gm, jm)
                for k in K_VALUES
                for gm in generator_models
                for jm in judge_models
            ]
            if completed and all(key in completed for key in group_keys):
                exp_counter += len(group_keys)
                outer.update(len(group_keys))
                continue

            # Build or load the vector store for this embed/chunk config
            if args.skip_index_rebuild and is_baseline:
                vectorstore = load_baseline_vectorstore()
            elif not csv_available and is_baseline:
                vectorstore = load_baseline_vectorstore()
            else:
                persist_dir = os.path.join(TEMP_INDEX_BASE, label)
                vectorstore = build_vectorstore(
                    label, embedding_model, chunk_size, chunk_overlap, persist_dir
                )

            for k in K_VALUES:
                retriever = vectorstore.as_retriever(search_kwargs={"k": k})

                for gen_model in generator_models:
                    llm = ChatOllama(model=gen_model, temperature=0, base_url=OLLAMA_BASE_URL)

                    for judge_model in judge_models:
                        exp_id = f"exp{exp_counter:03d}"
                        exp_key = (embedding_model, chunk_size, chunk_overlap, k, gen_model, judge_model)

                        if exp_key in completed:
                            exp_counter += 1
                            outer.update(1)
                            outer.set_postfix({"skipped": exp_id})
                            continue

                        judge_llm = ChatOllama(model=judge_model, temperature=0, base_url=OLLAMA_BASE_URL)

                        exp_meta = {
                            "exp_id": exp_id,
                            "embedding_model": embedding_model,
                            "chunk_size": chunk_size,
                            "chunk_overlap": chunk_overlap,
                            "k": k,
                            "generator_model": gen_model,
                            "judge_model": judge_model,
                            "timestamp": datetime.datetime.now().isoformat(),
                        }

                        desc = f"{exp_id} [{label}|k={k}|{gen_model}]"
                        rows = []
                        exp_start_wall = time.time()
                        exp_start_energy_uj = _read_rapl_uj()
                        exp_start_cpu_s = _process_cpu_time_s()

                        for item in tqdm(
                            QUESTION_SET,
                            desc=desc,
                            leave=False,
                            position=1,
                            unit="q",
                        ):
                            q = item["question"]
                            docs = augmented_retrieve(vectorstore, retriever, q)
                            raw_context = "\n\n".join(doc.page_content for doc in docs)
                            rag_prompt = RAG_PROMPT.format(context=raw_context, question=q)

                            q_start_energy_uj = _read_rapl_uj()
                            q_start_cpu_s = _process_cpu_time_s()

                            t0 = time.time()
                            raw_answer = llm.invoke(rag_prompt).content.strip()
                            gen_latency = round(time.time() - t0, 2)

                            t1 = time.time()
                            grounding = run_judge(judge_llm, GROUNDING_PROMPT, q, raw_context, raw_answer)
                            relevance = run_judge(judge_llm, RELEVANCE_PROMPT, q, raw_context, raw_answer)
                            judge_latency = round(time.time() - t1, 2)

                            q_total_latency = round(gen_latency + judge_latency, 2)
                            q_cpu_time_s = round(_process_cpu_time_s() - q_start_cpu_s, 3)
                            q_energy_j = _energy_delta_j(q_start_energy_uj, _read_rapl_uj())

                            rows.append({
                                **exp_meta,
                                "id": item["id"],
                                "question": item["question"],
                                "expected_answer": item["expected_answer"],
                                "answer_type": item["answer_type"],
                                "difficulty": item["difficulty"],
                                "course_scope": (
                                    ", ".join(item["course_scope"])
                                    if isinstance(item.get("course_scope"), list)
                                    else str(item.get("course_scope", ""))
                                ),
                                "evidence_fields": ", ".join(item.get("evidence_fields", [])),
                                "is_unanswerable": item.get("is_unanswerable", False),
                                "retrieved_docs": len(docs),
                                "context": clean_text(raw_context),
                                "answer": clean_text(raw_answer),
                                "generation_latency_s": gen_latency,
                                "judge_latency_s": judge_latency,
                                "total_latency_s": q_total_latency,
                                "cpu_time_s": q_cpu_time_s,
                                "energy_j": q_energy_j,
                                "grounding_score": normalize_score(grounding.get("score")),
                                "grounding_verdict": grounding.get("verdict", ""),
                                "grounding_reason": clean_text(grounding.get("reason", "")),
                                "unsupported_claims": json.dumps(
                                    grounding.get("unsupported_claims", []), ensure_ascii=False
                                ),
                                "relevance_score": normalize_score(relevance.get("score")),
                                "relevance_verdict": relevance.get("verdict", ""),
                                "relevance_reason": clean_text(relevance.get("reason", "")),
                                "missing_points": json.dumps(
                                    relevance.get("missing_points", []), ensure_ascii=False
                                ),
                            })

                        append_to_csv(rows, args.output)

                        exp_duration_s = round(time.time() - exp_start_wall, 2)
                        exp_energy_j = _energy_delta_j(exp_start_energy_uj, _read_rapl_uj())
                        exp_cpu_time_s = round(_process_cpu_time_s() - exp_start_cpu_s, 3)

                        grounding_passes = sum(1 for r in rows if r["grounding_verdict"] == "pass")
                        grounding_fails = sum(1 for r in rows if r["grounding_verdict"] == "fail")
                        relevance_passes = sum(1 for r in rows if r["relevance_verdict"] == "pass")
                        relevance_fails = sum(1 for r in rows if r["relevance_verdict"] == "fail")
                        n = len(rows)
                        append_to_csv([{
                            **exp_meta,
                            "total_questions": n,
                            "experiment_duration_s": exp_duration_s,
                            "total_cpu_time_s": exp_cpu_time_s,
                            "total_energy_j": exp_energy_j,
                            "mean_latency_per_question_s": round(sum(r["total_latency_s"] for r in rows) / n, 2),
                            "mean_cpu_time_per_question_s": round(exp_cpu_time_s / n, 3),
                            "mean_energy_per_question_j": round(exp_energy_j / n, 4) if exp_energy_j is not None else None,
                            "grounding_passes": grounding_passes,
                            "grounding_fails": grounding_fails,
                            "relevance_passes": relevance_passes,
                            "relevance_fails": relevance_fails,
                            "mean_grounding_score": round(sum(r["grounding_score"] for r in rows) / n, 3),
                            "mean_relevance_score": round(sum(r["relevance_score"] for r in rows) / n, 3),
                        }], summary_path)

                        exp_counter += 1
                        outer.update(1)
                        outer.set_postfix({
                            "last": exp_id,
                            "gnd": f"{grounding_passes}p/{grounding_fails}f",
                            "rel": f"{relevance_passes}p/{relevance_fails}f",
                        })

    # ── Cleanup temp indices ──────────────────────────────────────────────────
    if Path(TEMP_INDEX_BASE).exists():
        shutil.rmtree(TEMP_INDEX_BASE)
        print(f"\nCleaned up temporary indices from {TEMP_INDEX_BASE}")

    print(f"\nAll done.")
    print(f"  Per-question results : {args.output}")
    print(f"  Per-experiment summary: {summary_path}")

    df_summary = pd.read_csv(summary_path)
    print("\nExperiment summary (passes / fails):")
    print(df_summary[["exp_id", "embedding_model", "chunk_size", "k", "generator_model",
                       "grounding_passes", "grounding_fails",
                       "relevance_passes", "relevance_fails",
                       "mean_grounding_score", "mean_relevance_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
