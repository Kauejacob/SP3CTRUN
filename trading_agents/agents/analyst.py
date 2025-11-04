"""
Agente Analista Fundamental.
Coleta dados, valida, calcula score e gera relatório estruturado.
"""

# ============ IMPORTS E CONFIGURAÇÃO DE PATH ============
import sys
import os

# Adiciona raiz do projeto ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============ CARREGA VARIÁVEIS DE AMBIENTE ============
# Carrega .env da raiz do projeto
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# Carrega o .env de forma robusta
env_path = find_dotenv(usecwd=True)  # procura a partir do CWD do processo
if not env_path:  # se não encontrou, force o caminho relativo ao arquivo atual
    env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

# Valida se a API key foi carregada
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY não encontrada no .env!")

# Remove espaços
OPENAI_API_KEY = OPENAI_API_KEY.strip()

# ============ IMPORTS DO PROJETO ============
import json
from typing import Optional, Dict, Any

from agno.agent import Agent
from agno.models.openai import OpenAIChat

from models.schemas import FundamentalReport, Verdict
from data.yfinance_utils import get_fundamental_snapshot

# ============ FUNÇÕES AUXILIARES ============

def safe_get(data: Dict, key: str, default: Any = None) -> Any:
    """
    Acessa chave do dict de forma segura, tratando None.
    """
    value = data.get(key, default)
    if value is None:
        return default
    return value


def calculate_valuation_score(snapshot: Dict) -> float:
    """
    Calcula score de valuation (0-40 pontos).
    Empresa barata = score alto.
    """
    score = 0.0
    
    # P/E Ratio (15 pontos)
    pe = safe_get(snapshot, 'pe', None)
    if pe and pe > 0:
        if pe < 8:
            score += 15
        elif pe < 12:
            score += 12
        elif pe < 15:
            score += 8
        elif pe < 20:
            score += 4
    
    # P/B Ratio (10 pontos)
    pb = safe_get(snapshot, 'pb', None)
    if pb and pb > 0:
        if pb < 1.0:
            score += 10
        elif pb < 2.0:
            score += 7
        elif pb < 3.0:
            score += 4
    
    # P/S Ratio (10 pontos)
    ps = safe_get(snapshot, 'ps', None)
    if ps and ps > 0:
        if ps < 1.0:
            score += 10
        elif ps < 2.0:
            score += 7
        elif ps < 3.0:
            score += 4
    
    # Dividend Yield (5 pontos)
    dy = safe_get(snapshot, 'dividend_yield', None)
    if dy and dy > 0:
        if dy > 0.06:
            score += 5
        elif dy > 0.04:
            score += 3
        elif dy > 0.02:
            score += 1
    
    return min(score, 40.0)


def calculate_quality_score(snapshot: Dict) -> float:
    """
    Calcula score de qualidade (0-40 pontos).
    Empresa lucrativa e eficiente = score alto.
    """
    score = 0.0
    
    # Margem Líquida (15 pontos) - TRATAMENTO ROBUSTO
    net_margin = safe_get(snapshot, 'net_margin', None)
    if net_margin is not None:
        try:
            net_margin = float(net_margin)
            if net_margin > 0.20:
                score += 15
            elif net_margin > 0.15:
                score += 12
            elif net_margin > 0.10:
                score += 8
            elif net_margin > 0.05:
                score += 4
        except (ValueError, TypeError):
            pass
    
    # ROE (15 pontos) - TRATAMENTO ROBUSTO
    roe = safe_get(snapshot, 'roe', None)
    if roe is not None:
        try:
            roe = float(roe)
            if roe > 0.20:
                score += 15
            elif roe > 0.15:
                score += 12
            elif roe > 0.10:
                score += 8
            elif roe > 0.05:
                score += 4
        except (ValueError, TypeError):
            pass
    
    # Margem Operacional (10 pontos) - TRATAMENTO ROBUSTO
    op_margin = safe_get(snapshot, 'op_margin', None)
    if op_margin is not None:
        try:
            op_margin = float(op_margin)
            if op_margin > 0.20:
                score += 10
            elif op_margin > 0.15:
                score += 7
            elif op_margin > 0.10:
                score += 4
        except (ValueError, TypeError):
            pass
    
    return min(score, 40.0)


def calculate_risk_score(snapshot: Dict) -> float:
    """
    Calcula score de risco (0-20 pontos).
    Empresa com baixo risco financeiro = score alto.
    """
    score = 20.0
    
    # Debt/Equity
    de = safe_get(snapshot, 'debt_to_equity', None)
    if de is not None:
        try:
            de = float(de)
            if de > 2.0:
                score -= 10
            elif de > 1.5:
                score -= 7
            elif de > 1.0:
                score -= 4
            elif de > 0.5:
                score -= 2
        except (ValueError, TypeError):
            pass
    
    # Current Ratio
    cr = safe_get(snapshot, 'current_ratio', None)
    if cr is not None:
        try:
            cr = float(cr)
            if cr < 0.5:
                score -= 10
            elif cr < 1.0:
                score -= 5
            elif cr < 1.5:
                score -= 2
        except (ValueError, TypeError):
            pass
    
    return max(score, 0.0)


def calculate_overall_score(
    valuation_score: float,
    quality_score: float,
    risk_score: float
) -> tuple:
    """
    Calcula score geral e determina veredito.
    """
    total = valuation_score + quality_score + risk_score
    
    non_zero_scores = sum([
        valuation_score > 0,
        quality_score > 0,
        risk_score > 0
    ])
    confidence = non_zero_scores / 3.0
    
    if total >= 75:
        verdict = Verdict.BUY
    elif total >= 55:
        verdict = Verdict.HOLD
    else:
        verdict = Verdict.SELL
    
    return total, confidence, verdict


# ============ PROMPT DO AGENTE ============

ANALYST_INSTRUCTIONS = """
Você é um **Analista Fundamentalista Sênior** especializado em ações brasileiras.

## SUA MISSÃO:
Analisar os dados fundamentalistas fornecidos e gerar um relatório estruturado.

## DADOS DISPONÍVEIS:
Você receberá um JSON com:
- Valuation: P/E, P/B, P/S
- Qualidade: Margens (bruta, operacional, líquida), ROE, ROA
- Risco: Dívida, Liquidez (current ratio)
- Crescimento: Revenue e Net Income YoY

**IMPORTANTE:** Alguns dados podem estar ausentes (null). Adapte sua análise aos dados disponíveis.

## FORMATO DE SAÍDA:
Retorne JSON seguindo o schema FundamentalReport:
```json
{
  "ticker": "PETR4.SA",
  "as_of": "2024-01-15",
  "verdict": "buy",
  "score": 72.5,
  "confidence": 0.85,
  "summary": "Resumo executivo de 2-3 frases",
  "rationale": [
    "Motivo 1 para a recomendação",
    "Motivo 2...",
    "..."
  ],
  "risks": [
    "Risco 1 identificado",
    "Risco 2...",
    "..."
  ],
  "snapshot": { }
}
```

## REGRAS:
1. Liste 3-5 rationale (pontos positivos)
2. Liste 2-4 risks (pontos de atenção)
3. Seja objetivo e baseado em dados
4. Se dados críticos estão ausentes, mencione nos risks
5. JSON puro, sem markdown
"""

analyst_agent = Agent(
    name="FundamentalAnalyst",
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions=ANALYST_INSTRUCTIONS,
)


# ============ ORCHESTRATOR ============

def run_analyst(
    ticker: str,
    as_of: Optional[str] = None,
    verbose: bool = True
) -> Dict:
    """
    Executa análise fundamentalista completa.
    """
    if verbose:
        print(f"\n📊 Analisando {ticker}...")
    
    # 1. Coleta dados
    if verbose:
        print("   Coletando dados fundamentalistas...")
    
    try:
        snapshot = get_fundamental_snapshot(ticker, as_of)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Falha ao coletar dados: {e}",
            "ticker": ticker
        }
    
    # 2. Calcula scores
    if verbose:
        print("   Calculando scores...")
    
    valuation_score = calculate_valuation_score(snapshot)
    quality_score = calculate_quality_score(snapshot)
    risk_score = calculate_risk_score(snapshot)
    
    total_score, confidence, verdict = calculate_overall_score(
        valuation_score, quality_score, risk_score
    )
    
    if verbose:
        print(f"   Scores: Val={valuation_score:.1f} Qual={quality_score:.1f} Risk={risk_score:.1f}")
        print(f"   Total: {total_score:.1f}/100 → {verdict.value.upper()}")
    
    # 3. Prepara prompt
    snapshot_json = json.dumps(snapshot, indent=2, ensure_ascii=False)
    
    prompt = f"""
Analise os dados fundamentalistas abaixo e gere o relatório.

**Ticker:** {ticker}
**Data:** {snapshot['as_of']}
**Scores calculados:**
- Valuation: {valuation_score:.1f}/40
- Quality: {quality_score:.1f}/40
- Risk: {risk_score:.1f}/20
- **TOTAL: {total_score:.1f}/100**

**Veredito sugerido:** {verdict.value.upper()}
**Confiança:** {confidence:.0%}

**Dados brutos:**
```json
{snapshot_json}
```

Gere o relatório em JSON seguindo o schema FundamentalReport.
Use os scores calculados e explique o racional da recomendação.
"""
    
    if verbose:
        print("   Gerando relatório via LLM...")
    
    # 4. Chama LLM
    try:
        response = analyst_agent.run(prompt)
        content = str(response.content)
        
        # Remove markdown
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        # Parse JSON
        report_dict = json.loads(content)
        
        # Valida com Pydantic
        report = FundamentalReport(**report_dict)
        
        if verbose:
            print(f"   ✅ Relatório gerado: {report.verdict.value.upper()} "
                  f"(score: {report.score:.1f}, conf: {report.confidence:.0%})")
        
        return {
            "status": "success",
            "ticker": ticker,
            "report": report,
            "score": total_score,
            "confidence": confidence
        }
        
    except Exception as e:
        if verbose:
            print(f"   ❌ Erro ao gerar relatório: {e}")
            print("   ⚠️ Usando relatório simplificado (sem LLM)")
        
        # Fallback: relatório sem LLM
        report = FundamentalReport(
            ticker=ticker,
            as_of=snapshot['as_of'],
            verdict=verdict,
            score=total_score,
            confidence=confidence,
            summary=f"Análise automática: Score {total_score:.0f}/100 indica {verdict.value}",
            rationale=[
                f"Score de valuation: {valuation_score:.1f}/40",
                f"Score de qualidade: {quality_score:.1f}/40",
                f"Score de risco: {risk_score:.1f}/20"
            ],
            risks=[
                "Análise baseada em dados disponíveis",
                "Alguns indicadores podem estar ausentes"
            ],
            snapshot=snapshot
        )
        
        return {
            "status": "success",
            "ticker": ticker,
            "report": report,
            "score": total_score,
            "confidence": confidence
        }


# ============ TESTE ============

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analista Fundamental")
    parser.add_argument("--ticker", type=str, default="PETR4.SA")
    parser.add_argument("--as-of", type=str, help="Data YYYY-MM-DD")
    
    args = parser.parse_args()
    
    result = run_analyst(args.ticker, args.as_of, verbose=True)
    
    if result["status"] == "success":
        report = result["report"]
        
        print("\n" + "="*70)
        print("📋 RELATÓRIO FUNDAMENTALISTA")
        print("="*70)
        
        print(f"\n🎯 {report.ticker}")
        print(f"📅 {report.as_of}")
        print(f"💡 Veredito: {report.verdict.value.upper()}")
        print(f"📊 Score: {report.score:.1f}/100")
        print(f"🎲 Confiança: {report.confidence:.0%}")
        
        print(f"\n📝 Summary:")
        print(f"   {report.summary}")
        
        print(f"\n✅ Rationale:")
        for r in report.rationale:
            print(f"   • {r}")
        
        print(f"\n⚠️ Risks:")
        for risk in report.risks:
            print(f"   • {risk}")
        
        print("\n" + "="*70)
    else:

        print(f"\n❌ Erro: {result['message']}")
