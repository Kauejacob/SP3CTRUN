"""
Define o universo de investimento (30 ações mais líquidas da B3).
"""

import pandas as pd
import yfinance as yf
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import time
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


# ============ FUNÇÕES AUXILIARES ============

def _safe_download_single(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download robusto para UM ticker com tratamento de erros.
    """
    try:
        # Tenta download direto
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True
        )
        
        # CORREÇÃO: Verifica se é DataFrame válido ANTES de checar empty
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Download retornou tipo inválido: {type(df)}")
        
        # Agora pode checar empty com segurança
        if len(df) == 0:
            raise ValueError("DataFrame vazio após download")
        
        # Garante que tem coluna Close
        if 'Close' not in df.columns:
            if 'Adj Close' in df.columns:
                df['Close'] = df['Adj Close']
            else:
                raise ValueError("Sem coluna Close ou Adj Close")
        
        return df[['Close']].copy()
        
    except Exception as e:
        # Fallback: tenta via .history()
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(
                start=start_date,
                end=end_date,
                auto_adjust=True
            )
            
            if not isinstance(hist, pd.DataFrame):
                raise ValueError(f"History retornou tipo inválido: {type(hist)}")
            
            if len(hist) == 0:
                raise ValueError("DataFrame vazio após history()")
            
            if 'Close' not in hist.columns:
                if 'Adj Close' in hist.columns:
                    hist['Close'] = hist['Adj Close']
                else:
                    raise ValueError("Sem coluna Close ou Adj Close em history")
            
            return hist[['Close']].copy()
            
        except Exception as e2:
            raise RuntimeError(f"Ambos métodos falharam. Download: {e}, History: {e2}")


# ============ DEFINIÇÃO DO UNIVERSO ============

LIQUID_STOCKS_B3 = [
    # Petróleo & Gás
    "PETR3.SA", "PETR4.SA", "PRIO3.SA",
    
    # Mineração & Siderurgia
    "VALE3.SA", "CSNA3.SA", "GGBR4.SA",
    
    # Bancos
    "ITUB4.SA", "BBDC4.SA", "BBAS3.SA", "SANB11.SA", "BBDC3.SA",
    
    # Varejo
    "MGLU3.SA", "LREN3.SA", "AMER3.SA",
    
    # Energia Elétrica
    "ELET3.SA", "ELET6.SA", "CPFE3.SA", "CMIG4.SA",
    
    # Alimentos & Bebidas
    "ABEV3.SA", "BRFS3.SA",
    
    # Telecom
    "VIVT3.SA", "TIMS3.SA",
    
    # Papel & Celulose
    "SUZB3.SA",
    
    # Construção
    "CYRE3.SA", "MRVE3.SA",
    
    # Outros
    "WEGE3.SA", "RADL3.SA", "B3SA3.SA", "RENT3.SA", "EMBR3.SA"
]


def get_universe(
    start_date: str,
    end_date: str,
    min_data_points: int = 500
) -> List[str]:
    """
    Retorna o universo de ações válidas para o período.
    """
    print(f"\n📊 Validando universo de {len(LIQUID_STOCKS_B3)} ações...")
    print(f"   Período: {start_date} a {end_date}")
    print(f"   Mínimo de dados: {min_data_points} dias\n")
    
    valid_tickers = []
    failed_tickers = []
    
    for ticker in LIQUID_STOCKS_B3:
        try:
            # Baixa dados
            df = _safe_download_single(ticker, start_date, end_date)
            
            # Valida quantidade de dados
            num_points = len(df)
            if num_points < min_data_points:
                failed_tickers.append((ticker, f"Poucos dados ({num_points})"))
                print(f"   ⚠️ {ticker}: Dados insuficientes ({num_points} dias)")
                continue
            
            # Valida NaNs
            nan_count = int(df['Close'].isna().sum())
            nan_pct = nan_count / num_points
            
            if nan_pct > 0.1:  # Mais de 10% NaN
                failed_tickers.append((ticker, f"Muitos NaNs ({nan_pct:.1%})"))
                print(f"   ⚠️ {ticker}: Muitos dados faltantes ({nan_pct:.1%})")
                continue
            
            # Valida preços positivos
            positive_prices = int((df['Close'] > 0).sum())
            if positive_prices < num_points * 0.95:  # Pelo menos 95% válidos
                failed_tickers.append((ticker, "Preços inválidos"))
                print(f"   ⚠️ {ticker}: Preços inválidos")
                continue
            
            # Passou em todos os testes
            valid_tickers.append(ticker)
            print(f"   ✅ {ticker}: OK ({num_points} dias, {nan_pct:.1%} NaNs)")
            
        except Exception as e:
            error_msg = str(e)[:80]
            failed_tickers.append((ticker, error_msg))
            print(f"   ❌ {ticker}: {error_msg}")
    
    # Resumo
    print(f"\n{'='*70}")
    print(f"✅ Universo final: {len(valid_tickers)} ações válidas")
    print(f"❌ Excluídas: {len(failed_tickers)} ações")
    print(f"{'='*70}\n")
    
    if failed_tickers:
        print("Ações excluídas:")
        for ticker, reason in failed_tickers[:10]:  # Mostra apenas 10 primeiras
            print(f"  • {ticker}: {reason}")
        if len(failed_tickers) > 10:
            print(f"  ... e mais {len(failed_tickers) - 10} ações")
        print()
    
    return valid_tickers


def get_price_data(
    tickers: List[str],
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Baixa dados históricos de preços para múltiplos tickers.
    Retorna DataFrame com índice datetime e colunas = tickers.
    """
    print(f"\n📈 Baixando dados históricos para {len(tickers)} ações...")
    print(f"   Período: {start_date} a {end_date}")
    
    prices = pd.DataFrame()
    
    # Baixa um por um para maior controle
    for i, ticker in enumerate(tickers, 1):
        try:
            print(f"   [{i}/{len(tickers)}] {ticker}...", end=" ")
            
            df = _safe_download_single(ticker, start_date, end_date)
            prices[ticker] = df['Close']
            
            print("✅")
            
        except Exception as e:
            print(f"❌ {str(e)[:50]}")
    
    # Limpeza
    if len(prices.columns) > 0:
        # Remove dias onde TODAS as ações são NaN
        prices = prices.dropna(how='all')
        
        # Forward fill para feriados
        prices = prices.fillna(method='ffill')
        
        # Remove linhas ainda com NaN (início da série)
        prices = prices.dropna()
        
        print(f"\n✅ Dados carregados:")
        print(f"   Total de dias: {len(prices)}")
        if len(prices) > 0:
            print(f"   Data inicial: {prices.index[0].strftime('%Y-%m-%d')}")
            print(f"   Data final: {prices.index[-1].strftime('%Y-%m-%d')}")
        print(f"   Ações válidas: {len(prices.columns)}")
    else:
        print("\n❌ Nenhum dado foi carregado!")
    
    return prices


def get_ticker_info(ticker: str) -> Dict:
    """
    Retorna informações básicas de um ticker.
    """
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        
        return {
            'ticker': ticker,
            'name': info.get('longName', ticker),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'market_cap': info.get('marketCap', 0),
        }
    except Exception:
        return {
            'ticker': ticker,
            'name': ticker,
            'sector': 'Unknown',
            'industry': 'Unknown',
            'market_cap': 0,
        }


def print_universe_summary(tickers: List[str]):
    """
    Imprime resumo do universo de investimento agrupado por setor.
    """
    if not tickers:
        print("\n⚠️ Universo vazio, nada para resumir.")
        return
    
    print("\n" + "="*70)
    print("📋 RESUMO DO UNIVERSO DE INVESTIMENTO")
    print("="*70)
    
    # Agrupa por setor
    sectors = {}
    
    print("\nColetando informações dos tickers...")
    for ticker in tickers:
        info = get_ticker_info(ticker)
        sector = info['sector']
        
        if sector not in sectors:
            sectors[sector] = []
        sectors[sector].append(ticker)
    
    # Imprime por setor
    print()
    for sector, sector_tickers in sorted(sectors.items()):
        print(f"{sector}:")
        for ticker in sector_tickers:
            print(f"  • {ticker}")
        print()
    
    print(f"{'='*70}\n")


# ============ TESTE ============

if __name__ == "__main__":
    # Teste com período reduzido
    print("🧪 TESTE DO MÓDULO UNIVERSE")
    print("="*70)
    
    # Valida universo
    universe = get_universe(
        start_date="2022-01-01",  # Período menor para teste rápido
        end_date="2024-12-31",
        min_data_points=400  # Reduzido para 400 dias
    )
    
    if universe:
        print(f"\n✅ Universo validado com {len(universe)} ações:")
        for ticker in universe[:10]:  # Mostra apenas 10
            print(f"   • {ticker}")
        if len(universe) > 10:
            print(f"   ... e mais {len(universe) - 10} ações")
        
        # Testa download de preços (apenas 3 primeiras)
        print(f"\n🧪 Testando download de preços (3 ações)...")
        test_tickers = universe[:3]
        
        prices = get_price_data(
            test_tickers,
            "2023-01-01",
            "2024-01-01"
        )
        
        if not prices.empty:
            print(f"\n📊 Preview dos dados:")
            print(prices.head())
            print(f"\n...")
            print(prices.tail())
            
            # Estatísticas básicas
            print(f"\n📈 Estatísticas:")
            print(f"   Retorno médio diário:")
            returns = prices.pct_change().mean() * 100
            for ticker in test_tickers:
                print(f"      {ticker}: {returns[ticker]:.3f}%")
        
    else:
        print("\n❌ FALHA: Nenhuma ação válida no universo!")
        print("\nPossíveis causas:")
        print("  1. Conexão com internet instável")
        print("  2. yfinance com problemas temporários")
        print("  3. Tickers desatualizados/delisted")

        print("\nTente novamente em alguns minutos.")
