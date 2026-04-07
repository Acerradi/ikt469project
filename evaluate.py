from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
import json
import pandas as pd
from tqdm import tqdm

# 1) Load the same embedding model you used when indexing
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# 2) Load the persisted vector store
vectorstore = Chroma(
    persist_directory="./chroma_langchain_db",
    collection_name="uia_courses",
    embedding_function=embeddings,
)

# 3) Turn it into a retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 4) Load your Ollama LLM
llm = ChatOllama(
    model="qwen2.5:0.5b",   # or another model you actually pulled
    temperature=0,
    base_url="http://localhost:11434"
)
print("Loaded RAG system")

# LLM-as-Judge evaluation pipeline for your UiA course RAG system

import json
import pandas as pd
from tqdm import tqdm

from langchain_ollama import ChatOllama

# ----------------------------
# 1. Judge model
# ----------------------------
judge_llm = ChatOllama(
    model="qwen2.5:0.5b",  # upgrade later if needed
    temperature=0,
    base_url="http://localhost:11434"
)

# ----------------------------
# 2. Judge prompts
# ----------------------------
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
  "score": float,
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
  "score": float,
  "verdict": "pass" or "fail",
  "missing_points": [list],
  "reason": "short explanation"
}}

Rules:
- Penalize vague or incomplete answers
- Penalize irrelevant info
"""

# ----------------------------
# 3. Helper: robust JSON parser
# ----------------------------
def safe_parse_json(text: str):
    """
    Try to parse JSON from model output.
    Falls back to extracting substring between first { and last }.
    """
    try:
        return json.loads(text)
    except Exception:
        pass

    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end != -1:
            return json.loads(text[start:end])
    except Exception:
        pass

    return None

# ----------------------------
# 4. Judge runner
# ----------------------------
def run_judge(prompt_template, question, context, answer):
    prompt = prompt_template.format(
        question=question,
        context=context,
        answer=answer
    )

    response = judge_llm.invoke(prompt)
    parsed = safe_parse_json(response.content)

    if parsed is not None:
        return parsed

    return {
        "score": 0.0,
        "verdict": "fail",
        "reason": "Invalid JSON from judge",
    }

# ----------------------------
# 5. Full evaluation question set
# ----------------------------
question_set = [
    {
        "id": "q001",
        "question": "Which course is taught by Morten Goodwin?",
        "expected_answer": "IKT469 Deep Neural Networks (Spring 2026).",
        "answer_type": "factoid",
        "difficulty": "easy",
        "course_scope": ["IKT469"],
        "evidence_fields": ["course_leader", "title"]
    },
    {
        "id": "q002",
        "question": "How many ECTS credits is IKT469 Deep Neural Networks worth?",
        "expected_answer": "7.5 ECTS.",
        "answer_type": "factoid",
        "difficulty": "easy",
        "course_scope": ["IKT469"],
        "evidence_fields": ["title", "ects_credits"]
    },
    {
        "id": "q003",
        "question": "What is the teaching language of IKT112 Concepts of Machine Learning?",
        "expected_answer": "English.",
        "answer_type": "factoid",
        "difficulty": "easy",
        "course_scope": ["IKT112"],
        "evidence_fields": ["title", "teaching_language"]
    },
    {
        "id": "q004",
        "question": "Who is the course leader for IKT204 Data communication?",
        "expected_answer": "Sigurd Eskeland.",
        "answer_type": "factoid",
        "difficulty": "easy",
        "course_scope": ["IKT204"],
        "evidence_fields": ["title", "course_leader"]
    },
    {
        "id": "q005",
        "question": "In which semester is IKT443 WiFi and Internet of Things offered?",
        "expected_answer": "Spring.",
        "answer_type": "factoid",
        "difficulty": "easy",
        "course_scope": ["IKT443"],
        "evidence_fields": ["title", "lecture_semester"]
    },
    {
        "id": "q006",
        "question": "Is IKT115 Introduction to Artificial Intelligence Technology offered as a free-standing course?",
        "expected_answer": "No. It is available as a continuing education course (IKT902), but not as a free-standing course.",
        "answer_type": "factoid",
        "difficulty": "easy",
        "course_scope": ["IKT115"],
        "evidence_fields": ["title", "offered_as_a_free_standing_course"]
    },
    {
        "id": "q007",
        "question": "What is the duration of IKT302 Bachelor's Thesis, data?",
        "expected_answer": "½ year.",
        "answer_type": "factoid",
        "difficulty": "easy",
        "course_scope": ["IKT302"],
        "evidence_fields": ["title", "duration"]
    },
    {
        "id": "q008",
        "question": "Which course is led by Andreas Prinz?",
        "expected_answer": "IKT122 Success – AI Tools and Mindset for Goal Achievement (Spring 2026).",
        "answer_type": "factoid",
        "difficulty": "easy",
        "course_scope": ["IKT122"],
        "evidence_fields": ["course_leader", "title"]
    },
    {
        "id": "q009",
        "question": "What are the examination components for IKT112 Concepts of Machine Learning?",
        "expected_answer": "A 3-hour written exam worth 50% and a portfolio assessment worth 50%, with graded assessment.",
        "answer_type": "descriptive",
        "difficulty": "medium",
        "course_scope": ["IKT112"],
        "evidence_fields": ["title", "examinations"]
    },
    {
        "id": "q010",
        "question": "Summarize the main learning outcomes of IKT469 Deep Neural Networks.",
        "expected_answer": "Students should understand advanced deep learning concepts, transformer models, multimodal representation learning, generative models, explainability methods, advanced optimization, graph and sequence models, embeddings, retrieval-augmented systems, and be able to design, implement, evaluate, and critically assess advanced neural architectures.",
        "answer_type": "summary",
        "difficulty": "medium",
        "course_scope": ["IKT469"],
        "evidence_fields": ["title", "learning_outcomes"]
    },
    {
        "id": "q011",
        "question": "What teaching and learning methods are used in IKT460 Reinforcement Learning?",
        "expected_answer": "Combination of lectures, assignments, paper studies, lab work, report writing, and self-study. Tasks are done individually or in small groups of 2 students with group supervision.",
        "answer_type": "descriptive",
        "difficulty": "medium",
        "course_scope": ["IKT460"],
        "evidence_fields": ["title", "teaching_and_learning_methods"]
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
            "recommended_previous_knowledge"
        ]
    },
    {
        "id": "q013",
        "question": "What are the prerequisites for starting IKT302 Bachelor's Thesis, data?",
        "expected_answer": "The student must have passed at least 130 ECTS credits in their study plan by the start of the semester, and the topic and research question must be approved by the course leader for the bachelor thesis.",
        "answer_type": "descriptive",
        "difficulty": "medium",
        "course_scope": ["IKT302"],
        "evidence_fields": ["title", "prerequisites"]
    },
    {
        "id": "q014",
        "question": "What are the compulsory requirements for IKT590 Master's Thesis?",
        "expected_answer": "There must be 10 compulsory guidance meetings for every student/group, and the thesis press release must be submitted and approved.",
        "answer_type": "descriptive",
        "difficulty": "medium",
        "course_scope": ["IKT590"],
        "evidence_fields": ["title", "examination_requirements"]
    },
    {
        "id": "q015",
        "question": "Compare the assessment methods of IKT469 Deep Neural Networks and IKT468 Applied Algorithms.",
        "expected_answer": "IKT469 uses graded portfolio assessment only, with a postponed exam available for the portfolio. IKT468 uses a portfolio with project assignments worth 60% and a 3-hour written examination worth 40%.",
        "answer_type": "comparison",
        "difficulty": "medium",
        "course_scope": ["IKT469", "IKT468"],
        "evidence_fields": ["title", "examinations"]
    },
    {
        "id": "q016",
        "question": "Which has more ECTS credits: IKT112 Concepts of Machine Learning or IKT215 Pattern Recognition?",
        "expected_answer": "IKT215 Pattern Recognition has more credits. IKT112 is 5 ECTS, while IKT215 is 10 ECTS.",
        "answer_type": "comparison",
        "difficulty": "easy",
        "course_scope": ["IKT112", "IKT215"],
        "evidence_fields": ["title", "ects_credits"]
    },
    {
        "id": "q017",
        "question": "Compare the workload of IKT112, IKT215, and IKT469.",
        "expected_answer": "IKT112 has approximately 135 hours of workload, while IKT215 and IKT469 each have approximately 200 hours.",
        "answer_type": "comparison",
        "difficulty": "medium",
        "course_scope": ["IKT112", "IKT215", "IKT469"],
        "evidence_fields": ["title", "teaching_and_learning_methods"]
    },
    {
        "id": "q018",
        "question": "Which course has a larger exam component: IKT112 Concepts of Machine Learning or IKT443 WiFi and Internet of Things?",
        "expected_answer": "IKT443 has the larger exam component. Its individual 4-hour written exam counts 75% of the final grade, while IKT112 has a 3-hour written exam worth 50%.",
        "answer_type": "comparison",
        "difficulty": "medium",
        "course_scope": ["IKT112", "IKT443"],
        "evidence_fields": ["title", "examinations"]
    },
    {
        "id": "q019",
        "question": "Compare IKT452 Computer Vision and IKT460 Reinforcement Learning in terms of course content focus.",
        "expected_answer": "IKT452 focuses on computer vision topics such as image formation, camera geometry, feature detection, matching, stereo, motion estimation, tracking, scene understanding, and deep neural networks for vision. IKT460 focuses on reinforcement learning topics such as Markov decision processes, bandits, policy iteration, dynamic programming, TD-learning, direct policy search, deep RL, and end-to-end RL pipelines.",
        "answer_type": "comparison",
        "difficulty": "medium",
        "course_scope": ["IKT452", "IKT460"],
        "evidence_fields": ["title", "contents"]
    },
    {
        "id": "q020",
        "question": "Which course is most directly focused on deep learning and retrieval-augmented systems?",
        "expected_answer": "IKT469 Deep Neural Networks, because it explicitly includes advanced deep learning, transformer models, multimodal learning, embeddings, and retrieval-augmented generation with external knowledge integration.",
        "answer_type": "recommendation",
        "difficulty": "medium",
        "course_scope": ["IKT469"],
        "evidence_fields": ["title", "learning_outcomes", "contents"]
    },
    {
        "id": "q021",
        "question": "Which course would best fit a student who wants to learn reinforcement learning specifically?",
        "expected_answer": "IKT460 Reinforcement Learning.",
        "answer_type": "recommendation",
        "difficulty": "easy",
        "course_scope": ["IKT460"],
        "evidence_fields": ["title", "contents", "learning_outcomes"]
    },
    {
        "id": "q022",
        "question": "Which course is the best match for learning computer vision with hands-on implementation?",
        "expected_answer": "IKT452 Computer Vision, because it covers both theoretical and practical aspects of computing with images and includes hands-on experience solving real-life vision problems.",
        "answer_type": "recommendation",
        "difficulty": "medium",
        "course_scope": ["IKT452"],
        "evidence_fields": ["title", "contents", "learning_outcomes"]
    },
    {
        "id": "q023",
        "question": "Which courses appear most relevant for a student interested in IoT networking and communication technologies?",
        "expected_answer": "IKT443 WiFi and Internet of Things, IKT458 5G and IoT: Advanced, and IKT520 Security in IoT and Machine-Type Communication. IKT204 Data communication is also relevant as a foundational networking course.",
        "answer_type": "synthesis",
        "difficulty": "hard",
        "course_scope": ["IKT443", "IKT458", "IKT520", "IKT204"],
        "evidence_fields": ["title", "contents", "learning_outcomes"]
    },
    {
        "id": "q024",
        "question": "If a student wants a course on intelligent data processing and large-scale data-driven decision making, which course fits best?",
        "expected_answer": "IKT453 Intelligent Data Management.",
        "answer_type": "recommendation",
        "difficulty": "easy",
        "course_scope": ["IKT453"],
        "evidence_fields": ["title", "contents", "learning_outcomes"]
    },
    {
        "id": "q025",
        "question": "Can a student take IKT204 Data communication as a free-standing course?",
        "expected_answer": "Yes, subject to availability or capacity.",
        "answer_type": "factoid",
        "difficulty": "easy",
        "course_scope": ["IKT204"],
        "evidence_fields": ["title", "offered_as_a_free_standing_course"]
    },
    {
        "id": "q026",
        "question": "What must students complete before they can take the exam in IKT443 WiFi and Internet of Things?",
        "expected_answer": "Compulsory assignments must be approved.",
        "answer_type": "factoid",
        "difficulty": "easy",
        "course_scope": ["IKT443"],
        "evidence_fields": ["title", "examination_requirements"]
    },
    {
        "id": "q027",
        "question": "What prior knowledge is recommended for IKT452 Computer Vision?",
        "expected_answer": "IKT213 - Machine Vision.",
        "answer_type": "factoid",
        "difficulty": "easy",
        "course_scope": ["IKT452"],
        "evidence_fields": ["title", "recommended_previous_knowledge"]
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
            "recommended_previous_knowledge"
        ]
    },
    {
        "id": "q029",
        "question": "What are the minimum requirements to start IKT590 Master's Thesis?",
        "expected_answer": "Students must have passed examinations with no less than 75 ECTS credits in the master programme at the beginning of the semester, and the thesis topic must be approved by the University of Agder.",
        "answer_type": "descriptive",
        "difficulty": "medium",
        "course_scope": ["IKT590"],
        "evidence_fields": ["title", "prerequisites"]
    },
    {
        "id": "q030",
        "question": "Who is the course leader for IKT433 Distributed and Big Data Systems?",
        "expected_answer": "The provided data does not specify a course leader for IKT433.",
        "answer_type": "unanswerable",
        "difficulty": "easy",
        "course_scope": ["IKT433"],
        "evidence_fields": ["title", "course_leader"],
        "is_unanswerable": True
    },
    {
        "id": "q031",
        "question": "What is the exact duration of IKT433 Distributed and Big Data Systems?",
        "expected_answer": "The provided data does not clearly specify the duration for IKT433.",
        "answer_type": "unanswerable",
        "difficulty": "easy",
        "course_scope": ["IKT433"],
        "evidence_fields": ["title", "duration"],
        "is_unanswerable": True
    },
    {
        "id": "q032",
        "question": "Which textbook is required for IKT469 Deep Neural Networks?",
        "expected_answer": "The provided data does not list any required textbook for IKT469.",
        "answer_type": "unanswerable",
        "difficulty": "easy",
        "course_scope": ["IKT469"],
        "evidence_fields": ["title"],
        "is_unanswerable": True
    },
    {
        "id": "q033",
        "question": "On which weekday are lectures for IKT460 held?",
        "expected_answer": "The provided data does not include lecture weekdays or schedule details for IKT460.",
        "answer_type": "unanswerable",
        "difficulty": "easy",
        "course_scope": ["IKT460"],
        "evidence_fields": ["title"],
        "is_unanswerable": True
    },
    {
        "id": "q034",
        "question": "What is the classroom number for IKT122?",
        "expected_answer": "The provided data does not include classroom information for IKT122.",
        "answer_type": "unanswerable",
        "difficulty": "easy",
        "course_scope": ["IKT122"],
        "evidence_fields": ["title"],
        "is_unanswerable": True
    },
    {
        "id": "q035",
        "question": "Which spring 2026 courses in the dataset explicitly mention machine learning in either the title, learning outcomes, or contents?",
        "expected_answer": "At minimum: IKT112 Concepts of Machine Learning, IKT115 Introduction to Artificial Intelligence Technology, IKT215 Pattern Recognition, IKT459 Embedded Sensors, Signal Processing and Machine Learning for Autonomous Systems, IKT469 Deep Neural Networks. IKT460 Reinforcement Learning is also clearly machine learning related even though its title is more specific.",
        "answer_type": "synthesis",
        "difficulty": "hard",
        "course_scope": ["IKT112", "IKT115", "IKT215", "IKT459", "IKT469", "IKT460"],
        "evidence_fields": ["title", "learning_outcomes", "contents"]
    },
    {
        "id": "q036",
        "question": "Which courses in the dataset are 30 ECTS thesis courses?",
        "expected_answer": "IKT302 Bachelor's Thesis, data; IKT523 Master's Thesis, Cyber Security; and IKT590 Master's Thesis.",
        "answer_type": "list",
        "difficulty": "easy",
        "course_scope": ["IKT302", "IKT523", "IKT590"],
        "evidence_fields": ["title", "ects_credits"]
    },
    {
        "id": "q037",
        "question": "Which courses are taught in English and are available as free-standing courses, subject to availability?",
        "expected_answer": "Examples include IKT112, IKT215, IKT433, IKT443, IKT449, IKT452, IKT453, IKT458, IKT459, IKT460, IKT462, and IKT468, based on the provided records stating English teaching language and 'Yes' or equivalent wording for free-standing availability.",
        "answer_type": "synthesis",
        "difficulty": "hard",
        "course_scope": ["multiple"],
        "evidence_fields": ["title", "teaching_language", "offered_as_a_free_standing_course"]
    },
    {
        "id": "q038",
        "question": "Which courses explicitly require compulsory assignments or exercises to be approved before examination?",
        "expected_answer": "At least IKT204, IKT433, IKT443, IKT449, IKT519, and IKT520 explicitly state approval of compulsory exercises, assignments, hand-ins, or presentations as a requirement before the exam or examination.",
        "answer_type": "synthesis",
        "difficulty": "hard",
        "course_scope": ["IKT204", "IKT433", "IKT443", "IKT449", "IKT519", "IKT520"],
        "evidence_fields": ["title", "examination_requirements"]
    },
    {
        "id": "q039",
        "question": "Which courses appear most relevant for a student interested in cybersecurity?",
        "expected_answer": "IKT449 Selected Security Topics, IKT519 Blockchain and distributed ledger technology, IKT520 Security in IoT and Machine-Type Communication, and IKT523 Master's Thesis, Cyber Security. IKT204 also includes operational security and basic cryptography as supporting knowledge.",
        "answer_type": "synthesis",
        "difficulty": "hard",
        "course_scope": ["IKT449", "IKT519", "IKT520", "IKT523", "IKT204"],
        "evidence_fields": ["title", "contents", "learning_outcomes"]
    },
    {
        "id": "q040",
        "question": "Which course best matches a student wanting to build multimodal and retrieval-augmented systems using embeddings and external knowledge?",
        "expected_answer": "IKT469 Deep Neural Networks.",
        "answer_type": "recommendation",
        "difficulty": "medium",
        "course_scope": ["IKT469"],
        "evidence_fields": ["title", "learning_outcomes", "contents"]
    }
]

# ----------------------------
# 6. RAG evaluation loop
# ----------------------------
results = []

for item in tqdm(question_set, desc="Evaluating RAG"):
    qid = item["id"]
    q = item["question"]
    expected_answer = item["expected_answer"]
    answer_type = item["answer_type"]
    difficulty = item["difficulty"]
    course_scope = item.get("course_scope", [])
    evidence_fields = item.get("evidence_fields", [])
    is_unanswerable = item.get("is_unanswerable", False)

    # Retrieve relevant chunks
    docs = retriever.invoke(q)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Ask your RAG model to answer using only retrieved context
    rag_prompt = f"""
Answer the question using only the retrieved context.
If the answer is not in the context, say that the information is not provided.

Context:
{context}

Question:
{q}
"""
    answer = llm.invoke(rag_prompt).content

    # Judge grounding and relevance
    grounding = run_judge(GROUNDING_PROMPT, q, context, answer)
    relevance = run_judge(RELEVANCE_PROMPT, q, context, answer)

    results.append({
        "id": qid,
        "question": q,
        "expected_answer": expected_answer,
        "answer_type": answer_type,
        "difficulty": difficulty,
        "course_scope": ", ".join(course_scope) if isinstance(course_scope, list) else str(course_scope),
        "evidence_fields": ", ".join(evidence_fields),
        "is_unanswerable": is_unanswerable,
        "retrieved_docs": len(docs),
        "context": context,
        "answer": answer,
        "grounding_score": grounding.get("score"),
        "grounding_verdict": grounding.get("verdict"),
        "grounding_reason": grounding.get("reason"),
        "unsupported_claims": json.dumps(grounding.get("unsupported_claims", []), ensure_ascii=False),
        "relevance_score": relevance.get("score"),
        "relevance_verdict": relevance.get("verdict"),
        "relevance_reason": relevance.get("reason"),
        "missing_points": json.dumps(relevance.get("missing_points", []), ensure_ascii=False),
    })

# ----------------------------
# 7. Save results
# ----------------------------
df = pd.DataFrame(results)
df.to_csv("rag_evaluation.csv", index=False)

print("\nSaved results to rag_evaluation.csv")
print(df[[
    "id",
    "question",
    "grounding_score",
    "relevance_score",
    "grounding_verdict",
    "relevance_verdict"
]].head())
