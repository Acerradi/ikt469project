#!/usr/bin/env python3
"""
generate_expected_answers.py

Generates gold-standard expected answers for every evaluation question by
giving qwen2.5:14b the *complete* course catalogue as context (fetched
directly from ChromaDB), so answers are not limited by RAG top-k retrieval.

Requires:
    Ollama running with qwen2.5:14b pulled (uses GPU automatically)

Output:
    expected_answers.csv  —  question_id, question, expected_answer
"""

import csv
import os
from pathlib import Path

import pandas as pd
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from tqdm import tqdm

OLLAMA_BASE_URL = "http://localhost:11434"

CHROMA_DIR = "./chroma_langchain_db"
OUTPUT_PATH = "./expected_answers.csv"

QUESTION_SET = [
    {"id": "q001", "question": "Which course is taught by Morten Goodwin?"},
    {"id": "q002", "question": "How many ECTS credits is IKT469 Deep Neural Networks worth?"},
    {"id": "q003", "question": "What is the teaching language of IKT112 Concepts of Machine Learning?"},
    {"id": "q004", "question": "Who is the course leader for IKT204 Data communication?"},
    {"id": "q005", "question": "In which semester is IKT443 WiFi and Internet of Things offered?"},
    {"id": "q006", "question": "Is IKT115 Introduction to Artificial Intelligence Technology offered as a free-standing course?"},
    {"id": "q007", "question": "What is the duration of IKT302 Bachelor's Thesis, data?"},
    {"id": "q008", "question": "Which course is led by Andreas Prinz?"},
    {"id": "q009", "question": "What are the examination components for IKT112 Concepts of Machine Learning?"},
    {"id": "q010", "question": "Summarize the main learning outcomes of IKT469 Deep Neural Networks."},
    {"id": "q011", "question": "What teaching and learning methods are used in IKT460 Reinforcement Learning?"},
    {"id": "q012", "question": "What are the admission requirements and recommended previous knowledge for IKT204 Data communication?"},
    {"id": "q013", "question": "What are the prerequisites for starting IKT302 Bachelor's Thesis, data?"},
    {"id": "q014", "question": "What are the compulsory requirements for IKT590 Master's Thesis?"},
    {"id": "q015", "question": "Compare the assessment methods of IKT469 Deep Neural Networks and IKT468 Applied Algorithms."},
    {"id": "q016", "question": "Which has more ECTS credits: IKT112 Concepts of Machine Learning or IKT215 Pattern Recognition?"},
    {"id": "q017", "question": "Compare the workload of IKT112, IKT215, and IKT469."},
    {"id": "q018", "question": "Which course has a larger exam component: IKT112 Concepts of Machine Learning or IKT443 WiFi and Internet of Things?"},
    {"id": "q019", "question": "Compare IKT452 Computer Vision and IKT460 Reinforcement Learning in terms of course content focus."},
    {"id": "q020", "question": "Which course is most directly focused on deep learning and retrieval-augmented systems?"},
    {"id": "q021", "question": "Which course would best fit a student who wants to learn reinforcement learning specifically?"},
    {"id": "q022", "question": "Which course is the best match for learning computer vision with hands-on implementation?"},
    {"id": "q023", "question": "Which courses appear most relevant for a student interested in IoT networking and communication technologies?"},
    {"id": "q024", "question": "If a student wants a course on intelligent data processing and large-scale data-driven decision making, which course fits best?"},
    {"id": "q025", "question": "Can a student take IKT204 Data communication as a free-standing course?"},
    {"id": "q026", "question": "What must students complete before they can take the exam in IKT443 WiFi and Internet of Things?"},
    {"id": "q027", "question": "What prior knowledge is recommended for IKT452 Computer Vision?"},
    {"id": "q028", "question": "What prior knowledge is recommended for IKT519 Blockchain and distributed ledger technology?"},
    {"id": "q029", "question": "What are the minimum requirements to start IKT590 Master's Thesis?"},
    {"id": "q030", "question": "Who is the course leader for IKT433 Distributed and Big Data Systems?"},
    {"id": "q031", "question": "What is the exact duration of IKT433 Distributed and Big Data Systems?"},
    {"id": "q032", "question": "Which textbook is required for IKT469 Deep Neural Networks?"},
    {"id": "q033", "question": "On which weekday are lectures for IKT460 held?"},
    {"id": "q034", "question": "What is the classroom number for IKT122?"},
    {"id": "q035", "question": "Which spring 2026 courses in the dataset explicitly mention machine learning in either the title, learning outcomes, or contents?"},
    {"id": "q036", "question": "Which courses in the dataset are 30 ECTS thesis courses?"},
    {"id": "q037", "question": "Which courses are taught in English and are available as free-standing courses, subject to availability?"},
    {"id": "q038", "question": "Which courses explicitly require compulsory assignments or exercises to be approved before examination?"},
    {"id": "q039", "question": "Which courses appear most relevant for a student interested in cybersecurity?"},
    {"id": "q040", "question": "Which course best matches a student wanting to build multimodal and retrieval-augmented systems using embeddings and external knowledge?"},
]

GENERATION_PROMPT = """\
You are an expert on UiA (University of Agder) IKT course offerings for Spring 2026.
Below is the complete course catalogue. Answer the question accurately and completely \
using ONLY the information provided. If the information is genuinely not present in the \
catalogue, say so explicitly.

Be concise but thorough: include every relevant fact, do not add information not in the \
catalogue, and do not speculate.

Course Catalogue:
{context}

Question: {question}

Answer:"""


def load_full_context(chroma_dir: str) -> str:
    """Retrieve every document stored in ChromaDB and concatenate them."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vs = Chroma(
        persist_directory=chroma_dir,
        collection_name="uia_courses",
        embedding_function=embeddings,
    )
    # get() with no filter returns the entire collection
    raw = vs._collection.get(include=["documents"])
    docs = raw.get("documents", [])
    print(f"Loaded {len(docs)} documents from ChromaDB")
    # Deduplicate by full content — ChromaDB may hold multiple copies if the
    # collection was populated more than once without clearing first.
    seen, unique = set(), []
    for d in docs:
        if d not in seen:
            seen.add(d)
            unique.append(d)
    print(f"  → {len(unique)} unique documents after deduplication")
    return "\n\n---\n\n".join(unique)


def main() -> None:
    print("Loading ChromaDB …")
    full_context = load_full_context(CHROMA_DIR)

    llm = ChatOllama(
        model="phi4:14b",
        temperature=0,
        base_url=OLLAMA_BASE_URL,
        num_predict=1024,
    )

    rows = []
    for item in tqdm(QUESTION_SET, desc="Generating expected answers"):
        prompt = GENERATION_PROMPT.format(
            context=full_context,
            question=item["question"],
        )
        answer = llm.invoke(prompt).content.strip()
        rows.append({
            "question_id": item["id"],
            "question": item["question"],
            "expected_answer": answer,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False, quoting=csv.QUOTE_ALL)
    print(f"\nSaved {len(df)} expected answers to {OUTPUT_PATH}")
    print(df[["question_id", "expected_answer"]].to_string(index=False))


if __name__ == "__main__":
    main()