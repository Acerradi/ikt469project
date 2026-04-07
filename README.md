Instructions:
1. Start docker container using ´´´docker-compose up -d´´´
2. Download model using ´´´docker exec -it ollama ollama pull qwen2.5:0.5b´´´ (One time)
3. Run collect_store_data.ipynb (One time)
4. Run main.ipynb

To do:
- Implement pipeline that uses transformers and hugging face (Pure python, no Ollama), or dockerized Ollama
- Change scraping to collect more and better data.
- Implement better chunking, embedding and retrieval.
- Find a more suitable LLM model.
- Implement a terminal chat with RAG chatbot.
- Implement evaluation pipeline of RAG chatbot using LLM as judge?