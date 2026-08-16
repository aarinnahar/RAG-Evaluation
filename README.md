# RAG Evaluation Dashboard 🚀

> 🚧 **Work in Progress:** Core benchmarking engine and evaluation logic are implemented. Currently refactoring the codebase for production readiness, optimizing benchmarking pipelines, and expanding API integration tests.

## Overview
Evaluating Retrieval-Augmented Generation (RAG) pipelines often relies on manual testing and guesswork. The **RAG Evaluation Dashboard** is an engineering tool designed to systematically benchmark different document chunking strategies based on quantitative **cost vs. latency trade-offs**. 

Instead of guessing chunk sizes and overlaps, engineers can run experiments to determine the optimal retrieval configuration for their specific document types and latency budgets.

---

## Key Features & Architecture
- **Chunking Strategy Comparison:** Benchmarks fixed-size, semantic, token, and recursive chunking approaches.
- **Trade-off Analytics:** Measures vector storage costs, API call overhead, and retrieval latency metrics per query.
- **Evaluation Engine:** Quantitative scoring of chunk context relevancy and retrieval precision.
- **Modular Pipeline:** Extensible design for integrating multiple vector stores and LLM providers.

---

## Tech Stack
- **Language:** Python 3.10+
- **LLM & RAG Orchestration:** LangChain / LangGraph 
- **Vector Database:** FAISS 

---

## Project Status & Roadmap

- [x] Core chunking engine implementation
- [x] Basic cost and latency tracking pipeline
- [ ] Refactoring modular architecture for production grade
- [ ] REST API integration via FastAPI
- [ ] Containerization (Docker) & automated deployment pipeline
- [ ] Comprehensive unit and integration test suite

