# Setup

## 1. Start Docker
```bash
docker-compose up -d
```

## 2. Pull models (one time)
```bash
docker exec -it ollama ollama pull phi4:14b
docker exec -it ollama ollama pull qwen2.5:7b
docker exec -it ollama ollama pull llama3.2:3b
docker exec -it ollama ollama pull qwen2.5:0.5b
```

## 3. Install Python dependencies (one time)
```bash
pip install -r requirements.txt
```

## 4. Collect and index course data (one time)

Run `collect_store_data.ipynb` to scrape, chunk, embed, and index UiA IKT course data into ChromaDB.

## 5. Run evaluation
```bash
python evaluate.py
```

Results are appended to `experiment_results.csv` and `experiment_results_summary.csv`.

## 6. Run systematic experiments
```bash
python experiment_runner.py                        # full run
python experiment_runner.py --output my.csv        # custom output path
python experiment_runner.py --skip-index-rebuild   # use existing index
python experiment_runner.py --resume               # resume interrupted run
```

## experiment_runner.py

Runs a grid of RAG pipeline experiments and saves results to CSV. Each experiment varies one or more of:

| Variable | Options |
|---|---|
| Embedding model | `all-MiniLM-L6-v2`, `all-mpnet-base-v2` |
| Chunk size / overlap | `500/100` (baseline), `250/50` (smaller) |
| k (retrieved docs) | `3`, `5`, `10` |
| Generator model | `llama3.2:3b`, `qwen2.5:0.5b` |

The script automatically skips any Ollama model that is not pulled.

**Output files** (appended after each experiment, so partial results are preserved on interruption):

- `experiment_results.csv` — one row per question per experiment, including:
  - Retrieved context, generated answer, grounding and relevance scores/verdicts
  - `generation_latency_s`, `judge_latency_s`, `total_latency_s` — time breakdown per question
  - `cpu_time_s` — process CPU seconds consumed for that question (always available)
  - `energy_j` — CPU package energy in joules via Intel RAPL (requires root; `null` otherwise)

- `experiment_results_summary.csv` — one row per experiment, with the key metrics:
  - `grounding_passes` / `grounding_fails`, `relevance_passes` / `relevance_fails`
  - `experiment_duration_s` — total wall-clock time for the experiment
  - `mean_latency_per_question_s` — average total latency per question
  - `total_cpu_time_s`, `mean_cpu_time_per_question_s`
  - `total_energy_j`, `mean_energy_per_question_j` (RAPL; `null` if unavailable)

**Enabling RAPL energy measurement** (one-time, Linux only, requires sudo):
```bash
sudo chmod o+r /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj
sudo chmod o+r /sys/class/powercap/intel-rapl/intel-rapl:0/max_energy_range_uj
```

If `uia_ikt_courses.csv` is not present, only experiments using the existing baseline index (`all-MiniLM-L6-v2`, chunk `500/100`) will run; the other embedding/chunk configurations are skipped.

## To do:
- ~~Implement pipeline that uses transformers and hugging face (Pure python, no Ollama), or dockerized Ollama~~
- ~~Change scraping to collect more and better data.~~
- ~~Implement better chunking, embedding and retrieval.~~
- ~~Implement evaluation pipeline of RAG chatbot using LLM as judge?~~
- ~~Systematic experiments across embedding models, chunk sizes, k-values, and generator models~~
- Find a more suitable LLM model.
- Implement a terminal chat with RAG chatbot.
