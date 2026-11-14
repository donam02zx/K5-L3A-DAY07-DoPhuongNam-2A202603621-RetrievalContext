"""Central configuration loaded from .env (secrets) plus fixed experiment constants."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CHUNKS_PATH = DATA_DIR / "day6_chunks_B.json"
BENCHMARK_PATH = DATA_DIR / "benchmark.json"

PROVIDER = os.getenv("PROVIDER", "gemini")
MODEL = os.getenv("MODEL", "gemini-3.5-flash-lite")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))
DATABASE_URL = os.getenv("DATABASE_URL", "")
DB_TABLE = os.getenv("DB_TABLE", "progit_chunks")
DAY6_STRATEGY = os.getenv("DAY6_STRATEGY", "B")

RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# USD per 1M tokens, used only to estimate cost from reported token usage.
PRICE_IN_PER_M = float(os.getenv("PRICE_IN_PER_M", "0.10"))
PRICE_OUT_PER_M = float(os.getenv("PRICE_OUT_PER_M", "0.40"))
# Pause between model calls to stay under free-tier requests-per-minute limits.
MIN_SECONDS_BETWEEN_CALLS = float(os.getenv("MIN_SECONDS_BETWEEN_CALLS", "4.5"))
