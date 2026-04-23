Instructions:
1. Start docker container using ```docker-compose up -d```
2. Download models (one time):
```
docker exec -it ollama ollama pull llama3.2:3b
docker exec -it ollama ollama pull qwen2.5:0.5b
```
3. Run `collect_store_data.ipynb` to scrape, chunk, embed, and index course data (one time)
4. Run `evaluate.py` for a single evaluation with the default configuration
5. Run `experiment_runner.py` to systematically evaluate multiple configurations

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

- `experiment_results.csv` — one row per question per experiment, with the retrieved context, generated answer, grounding score/verdict, and relevance score/verdict
- `experiment_results_summary.csv` — one row per experiment, with pass/fail counts and mean scores as the key metrics

```
python experiment_runner.py                        # full run
python experiment_runner.py --output my.csv        # custom output path
python experiment_runner.py --skip-index-rebuild   # use existing chroma_langchain_db
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