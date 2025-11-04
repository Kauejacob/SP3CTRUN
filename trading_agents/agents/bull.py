"""
Agente Otimista (Bull) - Analisa oportunidades e cenários positivos.
"""

# ============ IMPORTS E CONFIGURAÇÃO DE PATH ============
import sys
import os
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
from typing import Optional
from datetime import datetime

from agno.agent import Agent
from agno.models.openai import OpenAIChat

from models.schemas import BullPerspective, FundamentalReport, Verdict


# ============ PROMPT DO AGENTE BULL ============

BULL_INSTRUCTIONS = """
Você é um **Analista Bullish (Otimista) Sênior** com 20 anos de experiência em identificar oportunidades e potencial de valorização em empresas.

## SUA MISSÃO:
Analisar CONSTRUTIVAMENTE o relatório do analista fundamental e os dados da empresa, focando em:
- Catalisadores de crescimento
- Forças competitivas
- Oportunidades de mercado
- Sinais de melhoria
- Fatores que podem levar a ganhos

## PROTOCOLO DE ANÁLISE:

### 1. Analise os Dados Fornecidos
Você receberá:
- Relatório completo do Analista Fundamental
- Snapshot com dados financeiros brutos
- Score e subscores de valuation/quality/risk

### 2. Identifique Oportunidades (opportunities)
Liste 5-7 oportunidades/catalisadores CONCRETOS baseados nos dados:
- Se valuation barato: "P/E de X está Y% abaixo da média, indicando subvalorização"
- Se margens altas: "Margem líquida de X% é Z pontos acima do setor, demonstrando poder de precificação"
- Se baixa dívida: "D/E de X indica baixo risco e espaço para alavancagem estratégica"

**REGRAS:**
- Cite NÚMEROS EXATOS dos dados
- Cada opportunity deve ter evidência quantitativa
- Evite generalidades ("empresa boa" ❌) → seja específico ("ROE de 25% vs 15% do setor indica vantagem competitiva" ✅)

### 3. Cenário Otimista (best_case_scenario)
Construa uma narrativa do MELHOR CENÁRIO plausível (2-3 parágrafos):
- O que pode dar certo?
- Encadeamento de eventos positivos
- Impacto estimado no preço/fundamentals
- Baseie-se nos dados reais fornecidos

### 4. Probabilidades e Estimativas
- **upside_probability** (0-1): Quão provável é o cenário positivo?
  * 0.7-1.0: Altamente provável, fundamentos sólidos
  * 0.4-0.7: Moderadamente provável, alguns catalisadores
  * 0.0-0.4: Pouco provável, mas potencial existe

- **estimated_upside** (% positivo): Alta estimada no melhor caso
  * Ex: 35.2 significa ganho de 35.2%
  * Base em múltiplos setoriais, potencial de rerating

### 5. Recomendação
- **recommended_action**: BUY (se oportunidade clara) | HOLD (se moderado) | SELL (só se riscos superarem upside)
- **confidence** (0-1): Sua confiança na análise

### 6. Evidências do Analista
Liste 3-5 pontos ESPECÍFICOS do relatório do analista que suportam sua visão bullish.
Cite textualmente se possível.

### 7. Métricas-Chave Analisadas
Destaque as métricas que mais pesaram na análise:
```json
{
  "pe_ratio": 12.5,
  "roe": 0.22,
  "net_margin": 0.18,
  "revenue_growth_yoy": 0.15
}
```

## FORMATO DE SAÍDA:
Retorne JSON seguindo EXATAMENTE o schema BullPerspective.

## EXEMPLO:
```json
{
  "ticker": "XPTO4.SA",
  "as_of": "2024-03-29",
  "opportunities": [
    "P/E de 12.5x está 50% abaixo da média do setor de 25x, indicando forte subvalorização",
    "ROE de 22% está 47% acima da média setorial de 15%, indicando alta eficiência operacional",
    "Margem líquida de 18% é 5pp superior ao setor, demonstrando poder de precificação",
    "Crescimento de receita de 15% YoY acima do PIB indica ganho de market share",
    "D/E de 0.3x indica baixíssimo risco e espaço para M&A estratégico"
  ],
  "best_case_scenario": "No melhor cenário, a empresa continua ganhando market share com crescimento de 15% ao ano, sustentado por margens superiores (18% vs 13% do setor). O P/E atual de 12.5x está muito abaixo do potencial: com ROE de 22% e crescimento sustentável, a empresa merece múltiplo de 20x (ainda conservador vs histórico de 25x). O rerating de múltiplos + crescimento orgânico pode levar a valorização de 80%+ em 12-18 meses. Adicionalmente, com D/E baixo (0.3x), há espaço para aquisições que acelerem crescimento.",
  "upside_probability": 0.70,
  "estimated_upside": 65.0,
  "recommended_action": "buy",
  "confidence": 0.80,
  "evidence_from_analyst": [
    "Analista destacou 'ROE de 22% demonstra alta eficiência de capital'",
    "Score de quality foi 35/40, indicando empresa saudável",
    "Analista notou: 'Valuation atrativo com múltiplos comprimidos'"
  ],
  "key_metrics_analyzed": {
    "pe_ratio": 12.5,
    "roe": 0.22,
    "net_margin": 0.18,
    "revenue_growth_yoy": 0.15,
    "debt_to_equity": 0.3
  }
}
```

## REGRAS CRÍTICAS:
- Use APENAS dados fornecidos
- Cite números EXATOS
- Seja OTIMISTA mas REALISTA
- Cada afirmação deve ter evidência quantitativa
- JSON puro, sem markdown
"""


# ============ AGENTE ============

bull_agent = Agent(
    name="BullAnalyst",
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions=BULL_INSTRUCTIONS,
)


# ============ ORCHESTRATOR ============

def run_bull(
    analyst_report: FundamentalReport,
    verbose: bool = True
) -> BullPerspective:
    """
    Executa análise otimista baseada no relatório do analista.
    
    Args:
        analyst_report: Relatório do analista fundamental
        verbose: Se True, imprime progresso
    
    Returns:
        BullPerspective com análise otimista
    """
    
    if verbose:
        print(f"\n🐂 Analisando perspectiva BULLISH para {analyst_report.ticker}...")
    
    # Prepara contexto para o agente
    prompt = f"""
Analise os dados abaixo sob uma perspectiva OTIMISTA e identifique todas as oportunidades e catalisadores.

# RELATÓRIO DO ANALISTA FUNDAMENTAL

**Ticker:** {analyst_report.ticker}
**Data:** {analyst_report.as_of}
**Veredito do Analista:** {analyst_report.verdict.value.upper()}
**Score:** {analyst_report.score:.1f}/100 (confiança: {analyst_report.confidence:.0%})

**Summary:**
{analyst_report.summary}

**Rationale:**
{chr(10).join(f"  • {r}" for r in analyst_report.rationale)}

**Risks identificados:**
{chr(10).join(f"  • {r}" for r in analyst_report.risks)}

# DADOS FINANCEIROS BRUTOS

{json.dumps(analyst_report.snapshot, indent=2, ensure_ascii=False)}

---

Gere a análise bullish em JSON seguindo o schema BullPerspective.
Foque nas OPORTUNIDADES e no que pode dar CERTO.
"""
    
    if verbose:
        print("   Gerando análise otimista via LLM...")
    
    response = bull_agent.run(prompt)
    
    # Parse da resposta
    try:
        content = str(response.content)
        
        # Remove markdown
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        # Parse JSON
        bull_dict = json.loads(content)
        
        # Valida com Pydantic
        bull_perspective = BullPerspective(**bull_dict)
        
        if verbose:
            print(f"   ✅ Análise concluída: {bull_perspective.recommended_action.value.upper()}")
            print(f"      Upside: +{bull_perspective.estimated_upside:.1f}%")
            print(f"      Probabilidade: {bull_perspective.upside_probability:.0%}")
            print(f"      Confiança: {bull_perspective.confidence:.0%}")
        
        return bull_perspective
        
    except Exception as e:
        if verbose:
            print(f"   ❌ Erro ao parsear resposta: {e}")
        
        raise ValueError(
            f"Falha ao parsear resposta do agente Bull: {e}\n"
            f"Resposta bruta: {str(response.content)[:500]}"
        )


# ============ TESTE STANDALONE ============

if __name__ == "__main__":
    # Para testar, precisa de um relatório do analista
    print("⚠️ Este agente precisa de um FundamentalReport como input.")
    print("   Execute via orchestrator.py ou crie um report manualmente para teste.")
    
    # Exemplo de teste com dados mock:
    from models.schemas import FundamentalSnapshot
    
    mock_snapshot = {
        "ticker": "TEST4.SA",
        "as_of": "2024-03-29",
        "price": 50.0,
        "pe": 12.5,
        "roe": 0.22,
        "net_margin": 0.18,
        "revenue_growth_yoy": 0.15,
        "debt_to_equity": 0.3,
        "evidence": ["mock_data"]
    }
    
    mock_report = FundamentalReport(
        ticker="TEST4.SA",
        as_of="2024-03-29",
        verdict=Verdict.BUY,
        score=75.0,
        confidence=0.80,
        summary="Empresa com valuation atrativo e margens superiores",
        rationale=[
            "P/E de 12.5x abaixo da média",
            "ROE de 22% indica alta eficiência",
            "Margens superiores ao setor"
        ],
        risks=[
            "Risco macroeconômico",
            "Competição em alguns segmentos"
        ],
        snapshot=mock_snapshot
    )
    
    print("\n🧪 Testando com dados mock...")
    bull_result = run_bull(mock_report, verbose=True)
    
    print("\n" + "="*70)
    print("RESULTADO DA ANÁLISE BULLISH")
    print("="*70)
    print(f"\n🎯 Recomendação: {bull_result.recommended_action.value.upper()}")
    print(f"📈 Upside estimado: +{bull_result.estimated_upside:.1f}%")
    print(f"✅ Probabilidade: {bull_result.upside_probability:.0%}")
    
    print(f"\n🔍 Oportunidades:")
    for opp in bull_result.opportunities:
        print(f"   • {opp}")
    
    print(f"\n📖 Melhor Cenário:")

    print(f"   {bull_result.best_case_scenario}")
