@echo off
set PYTHONIOENCODING=utf-8
.venv\Scripts\python scripts\seed_corpus.py
.venv\Scripts\python scripts\gen_agent_queries.py
.venv\Scripts\python scripts\gen_spend.py
.venv\Scripts\python scripts\verify_lite.py
