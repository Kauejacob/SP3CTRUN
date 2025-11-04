"""
Configurações centralizadas do projeto.
"""

import os
from dotenv import load_dotenv

# ============ CARREGA .env ============
load_dotenv()

# ============ API KEYS ============
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError(
        "❌ OPENAI_API_KEY não encontrada!\n"
        "Crie um arquivo .env na raiz com:\n"
        "OPENAI_API_KEY=sk-proj-sua-chave-aqui\n"
    )

# ============ MODELOS LLM ============
ANALYST_MODEL = "gpt-4o-mini"
SENIOR_MODEL = "gpt-4o"

# ============ PARÂMETROS ============
MIN_CONFIDENCE = 0.7
MIN_SCORE_BUY = 66
MAX_SCORE_SELL = 34

# ============ BACKTEST ============
BACKTEST_UNIVERSE = [
    "PETR4.SA",
    "VALE3.SA",
    "ITUB4.SA",
    "BBDC4.SA",
    "ABEV3.SA",
]

BACKTEST_START = "2020-01-01"
BACKTEST_END = "2024-12-31"

# ============ DEBUG ============

print(f"✅ Config carregado. API Key: {OPENAI_API_KEY[:20]}...")
