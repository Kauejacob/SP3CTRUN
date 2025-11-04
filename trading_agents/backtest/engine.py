"""
Engine principal de backtest.
Coordena análise dos agentes + execução de trades + tracking de performance.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import time

from backtest.portifolio import Portfolio
from backtest.universe import get_universe, get_price_data
from backtest.metrics import get_cdi_data, align_cdi_to_portfolio, calculate_metrics

from orchestrador import run_trading_pipeline
from models.schemas import Verdict


class BacktestEngine:
    """
    Engine de backtest que executa estratégia multi-agente.
    """
    
    def __init__(
        self,
        initial_capital: float,
        start_date: str,
        end_date: str,
        rebalance_frequency: str = 'weekly',  # 'weekly', 'monthly'
        universe_tickers: Optional[List[str]] = None,
        commission_pct: float = 0.001,
        min_position_size: float = 0.02,  # 2%
        max_position_size: float = 0.10,  # 10%
        verbose: bool = True
    ):
        """
        Args:
            initial_capital: Capital inicial em R$
            start_date: Data início YYYY-MM-DD
            end_date: Data fim YYYY-MM-DD
            rebalance_frequency: Frequência de rebalanceamento
            universe_tickers: Lista de tickers (None = usa universo padrão)
            commission_pct: Taxa de corretagem
            min_position_size: Tamanho mínimo de posição (%)
            max_position_size: Tamanho máximo de posição (%)
            verbose: Se True, imprime progresso
        """
        self.initial_capital = initial_capital
        self.start_date = start_date
        self.end_date = end_date
        self.rebalance_frequency = rebalance_frequency
        self.commission_pct = commission_pct
        self.min_position_size = min_position_size
        self.max_position_size = max_position_size
        self.verbose = verbose
        
        # Inicializa portfólio
        self.portfolio = Portfolio(
            initial_capital=initial_capital,
            commission_pct=commission_pct,
            min_position_size=min_position_size,
            max_position_size=max_position_size
        )
        
        # Dados
        self.universe_tickers = universe_tickers
        self.price_data: Optional[pd.DataFrame] = None
        self.cdi_data: Optional[pd.DataFrame] = None
        
        # Tracking
        self.decisions_history: List[Dict] = []
        self.rebalance_dates: List[str] = []
        
    def prepare_data(self):
        """
        Prepara dados de preços e CDI.
        """
        if self.verbose:
            print("\n" + "="*70)
            print("🔧 PREPARANDO DADOS")
            print("="*70)
        
        # 1. Define universo
        if self.universe_tickers is None:
            if self.verbose:
                print("\nValidando universo de ações...")
            
            self.universe_tickers = get_universe(
                start_date=self.start_date,
                end_date=self.end_date,
                min_data_points=400
            )
            
            if len(self.universe_tickers) < 10:
                raise ValueError(
                    f"Universo muito pequeno ({len(self.universe_tickers)} ações). "
                    "Necessário pelo menos 10 ações."
                )
        
        # 2. Baixa preços
        if self.verbose:
            print(f"\nBaixando dados de preços para {len(self.universe_tickers)} ações...")
        
        self.price_data = get_price_data(
            self.universe_tickers,
            self.start_date,
            self.end_date
        )
        
        if self.price_data.empty:
            raise ValueError("Falha ao baixar dados de preços")
        
        # 3. Baixa CDI
        if self.verbose:
            print("\nBaixando dados do CDI...")
        
        self.cdi_data = get_cdi_data(self.start_date, self.end_date)
        
        if self.verbose:
            print(f"\n✅ Dados preparados:")
            print(f"   Ações: {len(self.price_data.columns)}")
            print(f"   Período: {self.price_data.index[0].strftime('%Y-%m-%d')} a "
                  f"{self.price_data.index[-1].strftime('%Y-%m-%d')}")
            print(f"   Dias: {len(self.price_data)}")
    
    # backtest/engine.py (SUBSTITUIR a função get_rebalance_dates completa)

    def get_rebalance_dates(self) -> List[pd.Timestamp]:
        """
        Gera datas de rebalanceamento baseado na frequência configurada.
        
        Returns:
            Lista de timestamps com as datas de rebalanceamento
        """
        if self.rebalance_frequency == 'weekly':
            # Toda segunda-feira
            all_dates = self.price_data.index
            mondays = [d for d in all_dates if d.weekday() == 0]
            return mondays
        
        elif self.rebalance_frequency == 'monthly':
            # Primeiro dia útil do mês
            all_dates = self.price_data.index
            first_days = []
            current_month = None
            
            for date in all_dates:
                if date.month != current_month:
                    first_days.append(date)
                    current_month = date.month
            
            return first_days
        
        elif self.rebalance_frequency == 'quarterly':
            # Primeiro dia útil de cada trimestre (Jan, Abr, Jul, Out)
            all_dates = self.price_data.index
            quarterly_dates = []
            current_quarter = None
            
            for date in all_dates:
                # Calcula trimestre (0=Jan-Mar, 1=Abr-Jun, 2=Jul-Set, 3=Out-Dez)
                quarter = (date.month - 1) // 3
                year_quarter = (date.year, quarter)
                
                if year_quarter != current_quarter:
                    quarterly_dates.append(date)
                    current_quarter = year_quarter
            
            return quarterly_dates
        
        else:
            raise ValueError(
                f"Frequência inválida: {self.rebalance_frequency}. "
                f"Use 'weekly', 'monthly' ou 'quarterly'"
            )
        
    def run_agents_for_ticker(
        self,
        ticker: str,
        as_of: str
    ) -> Optional[Dict]:
        """
        Executa pipeline de agentes para um ticker.
        
        Args:
            ticker: Ticker da ação
            as_of: Data de referência
        
        Returns:
            Dict com decisão ou None se falhou
        """
        try:
            state = run_trading_pipeline(
                ticker=ticker,
                as_of=as_of,
                verbose=False  # Silencioso para não poluir output
            )
            
            if state.pipeline_status == "completed" and state.senior_decision:
                return {
                    'ticker': ticker,
                    'verdict': state.senior_decision.final_verdict,
                    'position_size': state.senior_decision.position_size,
                    'confidence': state.senior_decision.confidence,
                    'stop_loss': state.senior_decision.stop_loss,
                    'take_profit': state.senior_decision.take_profit,
                    'analyst_score': state.analyst_report.score if state.analyst_report else None,
                }
            
            return None
            
        except Exception as e:
            if self.verbose:
                print(f"   ⚠️ {ticker}: Erro na análise - {str(e)[:50]}")
            return None
    
    # backtest/engine.py (SUBSTITUIR a função rebalance_portfolio INTEIRA)

    def rebalance_portfolio(self, date: pd.Timestamp):
        """
        Rebalanceamento agressivo com 3 camadas de qualidade.
        Meta: 85-95% sempre investido.
        """
        date_str = date.strftime('%Y-%m-%d')
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"🔄 REBALANCEAMENTO: {date_str}")
            print(f"{'='*70}")
        
        # Preços atuais
        current_prices = self.price_data.loc[date].to_dict()
        self.portfolio.update_prices(current_prices)
        
        # Aplica SELIC ao cash (13% ao ano = 0.035% ao dia)
        selic_daily = 0.00035
        self.portfolio.apply_selic_to_cash(date_str, selic_daily)
        
        if self.verbose:
            print(f"\n💰 Status atual:")
            print(f"   Cash: R$ {self.portfolio.cash:,.2f}")
            print(f"   Posições: R$ {self.portfolio.positions_value:,.2f}")
            print(f"   Exposição: {self.portfolio.exposure:.1f}%")
            print(f"   Total: R$ {self.portfolio.total_value:,.2f}")
        
        # ============ ANÁLISE DOS AGENTES ============
        
        decisions = []
        
        if self.verbose:
            print(f"\n🤖 Analisando {len(self.universe_tickers)} ações...")
        
        for i, ticker in enumerate(self.universe_tickers, 1):
            if self.verbose and i % 5 == 0:
                print(f"   Progresso: {i}/{len(self.universe_tickers)}")
            
            decision = self.run_agents_for_ticker(ticker, date_str)
            
            if decision:
                decisions.append(decision)
        
        if self.verbose:
            print(f"\n✅ {len(decisions)} análises concluídas")
        
        # ============ CLASSIFICAÇÃO POR QUALIDADE ============
        
        # Camada 1: BUY (score 65+)
        buy_signals = [d for d in decisions if d['verdict'] == Verdict.BUY]
        
        # Camada 2: HOLD (score 45-64)
        hold_signals = [d for d in decisions if d['verdict'] == Verdict.HOLD]
        
        # Ordena por qualidade (score * confidence)
        buy_signals.sort(
            key=lambda x: (x['analyst_score'] or 0) * (x['confidence'] or 0),
            reverse=True
        )
        hold_signals.sort(
            key=lambda x: (x['analyst_score'] or 0) * (x['confidence'] or 0),
            reverse=True
        )
        
        if self.verbose:
            print(f"\n📊 Classificação:")
            print(f"   🟢 BUY (score 65+): {len(buy_signals)}")
            print(f"   🟡 HOLD (score 45-64): {len(hold_signals)}")
        
        # ============ ESTRATÉGIA DE ALOCAÇÃO ============
        
        total_capital = self.portfolio.total_value
        
        # Calcula alocação ideal
        num_buy = len(buy_signals)
        num_hold = len(hold_signals)
        
        # Alocação dinâmica baseada na quantidade de sinais
        if num_buy >= 15:
            # Muitos BUY: 90% em BUY, 5% em HOLD, 5% cash
            buy_allocation = 0.90
            hold_allocation = 0.05
            cash_target = 0.05
        elif num_buy >= 10:
            # BUY bom: 80% em BUY, 10% em HOLD, 10% cash
            buy_allocation = 0.80
            hold_allocation = 0.10
            cash_target = 0.10
        elif num_buy >= 5:
            # BUY moderado: 60% em BUY, 25% em HOLD, 15% cash
            buy_allocation = 0.60
            hold_allocation = 0.25
            cash_target = 0.15
        else:
            # Poucos BUY: 40% em BUY, 45% em HOLD, 15% cash
            buy_allocation = 0.40
            hold_allocation = 0.45
            cash_target = 0.15
        
        if self.verbose:
            print(f"\n🎯 Alocação alvo:")
            print(f"   BUY: {buy_allocation*100:.0f}%")
            print(f"   HOLD: {hold_allocation*100:.0f}%")
            print(f"   Cash (SELIC): {cash_target*100:.0f}%")
        
        # ============ EXECUÇÃO: VENDE POSIÇÕES RUINS PRIMEIRO ============
        
        # Identifica posições a manter
        keep_tickers = set(
            [s['ticker'] for s in buy_signals[:15]] +  # Top 15 BUY
            [s['ticker'] for s in hold_signals[:20]]    # Top 20 HOLD
        )
        
        current_tickers = set(self.portfolio.positions.keys())
        to_sell = current_tickers - keep_tickers
        
        if to_sell and self.verbose:
            print(f"\n💸 Vendendo {len(to_sell)} posições sem sinal/ruins:")
        
        for ticker in to_sell:
            if ticker in current_prices:
                price = current_prices[ticker]
                trade = self.portfolio.sell(
                    ticker=ticker,
                    price=price,
                    date=date_str,
                    reason='NO_SIGNAL'
                )
                
                if trade and self.verbose:
                    print(f"   💸 SELL {trade.shares} {ticker} @ R${price:.2f}")
        
        # ============ EXECUÇÃO: COMPRA BUY SIGNALS ============
        
        if buy_signals and self.verbose:
            print(f"\n🟢 Comprando BUY signals (Top {min(15, len(buy_signals))}):")
        
        buy_capital = total_capital * buy_allocation
        per_buy_position = buy_capital / min(15, max(1, len(buy_signals)))
        buy_position_pct = (per_buy_position / total_capital) * 100
        
        for signal in buy_signals[:15]:
            ticker = signal['ticker']
            
            if ticker not in current_prices:
                continue
            
            price = current_prices[ticker]
            
            # Se já tem posição, apenas ajusta
            if ticker in self.portfolio.positions:
                continue  # Mantém a posição existente
            
            # Compra nova posição
            trade = self.portfolio.buy(
                ticker=ticker,
                price=price,
                target_pct=buy_position_pct,
                date=date_str,
                stop_loss=signal['stop_loss'],
                take_profit=signal['take_profit'],
                reason='REBALANCE'
            )
            
            if trade and self.verbose:
                print(f"   🟢 BUY {trade.shares} {ticker} @ R${price:.2f} "
                    f"({buy_position_pct:.1f}%, score: {signal['analyst_score']:.0f})")
        
        # ============ EXECUÇÃO: COMPRA HOLD SIGNALS ============
        
        if hold_signals and hold_allocation > 0 and self.verbose:
            print(f"\n🟡 Comprando HOLD signals (Top {min(20, len(hold_signals))}):")
        
        hold_capital = total_capital * hold_allocation
        per_hold_position = hold_capital / min(20, max(1, len(hold_signals)))
        hold_position_pct = (per_hold_position / total_capital) * 100
        
        for signal in hold_signals[:20]:
            ticker = signal['ticker']
            
            if ticker not in current_prices:
                continue
            
            # Se já tem posição (de BUY ou anterior), pula
            if ticker in self.portfolio.positions:
                continue
            
            price = current_prices[ticker]
            
            # Compra nova posição
            trade = self.portfolio.buy(
                ticker=ticker,
                price=price,
                target_pct=hold_position_pct,
                date=date_str,
                stop_loss=signal['stop_loss'],
                take_profit=signal['take_profit'],
                reason='HOLD_EXPOSURE'
            )
            
            if trade and self.verbose:
                print(f"   🟡 BUY {trade.shares} {ticker} @ R${price:.2f} "
                    f"({hold_position_pct:.1f}%, score: {signal['analyst_score']:.0f})")
        
        # ============ RESUMO FINAL ============
        
        if self.verbose:
            print(f"\n📊 Resumo pós-rebalanceamento:")
            print(f"   Exposição: {self.portfolio.exposure:.1f}%")
            print(f"   Posições: {self.portfolio.num_positions}")
            print(f"   Cash (rende SELIC): R$ {self.portfolio.cash:,.2f} ({(self.portfolio.cash/total_capital)*100:.1f}%)")
        
        # Salva decisões
        self.decisions_history.extend(decisions)
        self.rebalance_dates.append(date_str)
    
    def run(self):
        """
        Executa backtest completo.
        """
        start_time = time.time()
        
        if self.verbose:
            print("\n" + "="*70)
            print("🚀 INICIANDO BACKTEST")
            print("="*70)
            print(f"   Capital: R$ {self.initial_capital:,.2f}")
            print(f"   Período: {self.start_date} a {self.end_date}")
            print(f"   Rebalanceamento: {self.rebalance_frequency}")
        
        # 1. Prepara dados
        self.prepare_data()
        
        # 2. Gera datas de rebalanceamento
        rebalance_dates = self.get_rebalance_dates()
        
        if self.verbose:
            print(f"\n📅 {len(rebalance_dates)} datas de rebalanceamento")
        
        # 3. Loop principal
        for date in self.price_data.index:
            date_str = date.strftime('%Y-%m-%d')
            
            # Atualiza preços
            current_prices = self.price_data.loc[date].to_dict()
            self.portfolio.update_prices(current_prices)
            
            # Aplica SELIC diariamente ao cash
            self.portfolio.apply_selic_to_cash(date_str, 0.00035)
            
            # Verifica stops
            self.portfolio.check_stops(date_str)
            
            # Rebalanceia se for data de rebalanceamento
            if date in rebalance_dates:
                self.rebalance_portfolio(date)
            
            # Registra estado
            self.portfolio.record_state(date_str)
        
        # 4. Finaliza
        elapsed = time.time() - start_time
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"✅ BACKTEST CONCLUÍDO")
            print(f"{'='*70}")
            print(f"   Tempo: {elapsed/60:.1f} minutos")
            print(f"   Rebalanceamentos: {len(self.rebalance_dates)}")
            print(f"   Trades executados: {len(self.portfolio.trades)}")
    
    def get_results(self) -> Dict:
        """
        Retorna resultados do backtest.
        """
        # Histórico do portfólio
        history_df = self.portfolio.get_history_df()
        
        # Alinha CDI
        cdi_aligned = align_cdi_to_portfolio(history_df.index, self.cdi_data)
        
        # Calcula métricas
        metrics = calculate_metrics(history_df, cdi_aligned)
        
        # Trades
        trades_df = self.portfolio.get_trades_df()
        
        # Posições finais
        positions_df = self.portfolio.get_positions_summary()
        
        # Resumo
        summary = self.portfolio.summary()
        
        return {
            'metrics': metrics,
            'history': history_df,
            'cdi': cdi_aligned,
            'trades': trades_df,
            'positions': positions_df,
            'summary': summary,
            'rebalance_dates': self.rebalance_dates,
            'decisions': self.decisions_history
        }


# ============ TESTE ============

if __name__ == "__main__":
    print("🧪 TESTE DO BACKTEST ENGINE")
    print("="*70)
    
    # TESTE RÁPIDO: Período longo o suficiente + universo reduzido
    engine = BacktestEngine(
        initial_capital=10_000_000,  # 10M para teste (mais rápido)
        start_date="2022-01-01",  # ✅ 2 anos (suficiente para 400+ dias)
        end_date="2023-12-31",
        rebalance_frequency='monthly',  # ✅ Mensal = menos rebalanceamentos
        min_position_size=0.03,  # 3%
        max_position_size=0.15,  # 15%
        verbose=True
    )
    
    # ✅ FORÇA UNIVERSO PEQUENO PARA TESTE RÁPIDO
    print("\n⚠️ MODO TESTE: Usando apenas 5 ações para validação rápida")
    print("   (No backtest real, usará todas as 30 ações do universo)")
    
    engine.universe_tickers = [
        'PETR4.SA',
        'VALE3.SA', 
        'ITUB4.SA',
        'BBDC4.SA',
        'ABEV3.SA'
    ]
    
    try:
        # Executa
        engine.run()
        
        # Resultados
        results = engine.get_results()
        
        print("\n" + "="*70)
        print("📊 RESULTADOS DO TESTE")
        print("="*70)
        
        summary = results['summary']
        metrics = results['metrics']
        
        print(f"\n💰 Financeiro:")
        print(f"   Capital Inicial: R$ {summary['initial_capital']:,.2f}")
        print(f"   Valor Final: R$ {summary['current_value']:,.2f}")
        print(f"   Retorno: {summary['total_return_pct']:+.2f}%")
        
        print(f"\n📊 Performance:")
        print(f"   Retorno Anualizado: {metrics['annualized_return_pct']:+.2f}%")
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"   Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        
        print(f"\n📈 Operações:")
        print(f"   Total de trades: {summary['num_trades']}")
        print(f"   Posições finais: {summary['num_positions']}")
        print(f"   Rebalanceamentos: {len(engine.rebalance_dates)}")
        
        print("\n✅ Teste concluído com sucesso!")
        print("   Use run_backtest.py para backtest completo de 3 anos")
        
    except Exception as e:
        print(f"\n❌ ERRO NO TESTE: {e}")
        import traceback

        traceback.print_exc()
