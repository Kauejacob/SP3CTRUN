"""
Script principal para executar backtest completo do fundo multi-agente.

Uso:
    python run_backtest.py --capital 50000000 --start 2023-01-01 --end 2025-01-01
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
from datetime import datetime
import pandas as pd

from backtest.engine import BacktestEngine
from backtest.metrics import print_metrics
from backtest.visualization import create_performance_report

# run_backtest.py (continuação)

def main():
    parser = argparse.ArgumentParser(
        description="Backtest de Fundo Multi-Agente com R$ 50 Milhões"
    )
    
    parser.add_argument(
        '--capital',
        type=float,
        default=50_000_000,
        help='Capital inicial em R$ (padrão: 50M)'
    )
    
    parser.add_argument(
        '--start',
        type=str,
        default='2023-01-01',
        help='Data início YYYY-MM-DD (padrão: 2023-01-01)'
    )
    
    parser.add_argument(
        '--end',
        type=str,
        default='2025-09-01',
        help='Data fim YYYY-MM-DD (padrão: 2025-09-01)'
    )
    
    parser.add_argument(
        '--rebalance',
        type=str,
        default='monthly',
        choices=['weekly', 'monthly', 'quarterly'],
        help='Frequência de rebalanceamento (padrão: weekly)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='backtest_results',
        help='Diretório de saída (padrão: backtest_results)'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Não gera gráficos'
    )
    
    args = parser.parse_args()
    
    # ============ HEADER ============
    print("\n" + "="*70)
    print("🏦 BACKTEST DE FUNDO QUANTITATIVO MULTI-AGENTE")
    print("="*70)
    print(f"\n💰 Capital Inicial: R$ {args.capital:,.2f}")
    print(f"📅 Período: {args.start} a {args.end}")
    print(f"🔄 Rebalanceamento: {args.rebalance}")
    print(f"📁 Output: {args.output}/")
    
    # Cria diretório de output
    os.makedirs(args.output, exist_ok=True)
    
    # ============ INICIALIZA ENGINE ============
    engine = BacktestEngine(
        initial_capital=args.capital,
        start_date=args.start,
        end_date=args.end,
        rebalance_frequency=args.rebalance,
        commission_pct=0.001,  # 0.1%
        min_position_size=0.08,  # 8%
        max_position_size=0.15,  # 15%
        verbose=True
    )
    
    # ============ EXECUTA BACKTEST ============
    print("\n" + "="*70)
    print("🚀 EXECUTANDO BACKTEST")
    print("="*70)
    
    try:
        engine.run()
    except Exception as e:
        print(f"\n❌ ERRO DURANTE EXECUÇÃO: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ============ COLETA RESULTADOS ============
    print("\n" + "="*70)
    print("📊 PROCESSANDO RESULTADOS")
    print("="*70)
    
    results = engine.get_results()
    
    # ============ MÉTRICAS ============
    print_metrics(results['metrics'])
    
    # ============ RESUMO DO PORTFÓLIO ============
    print("\n" + "="*70)
    print("💼 RESUMO DO PORTFÓLIO")
    print("="*70)
    
    summary = results['summary']
    print(f"\n💵 Financeiro:")
    print(f"   Capital Inicial: R$ {summary['initial_capital']:,.2f}")
    print(f"   Valor Final: R$ {summary['current_value']:,.2f}")
    print(f"   Cash: R$ {summary['cash']:,.2f}")
    print(f"   Posições: R$ {summary['positions_value']:,.2f}")
    print(f"   P&L: R$ {summary['total_return_brl']:,.2f} ({summary['total_return_pct']:+.2f}%)")
    
    print(f"\n📊 Portfólio:")
    print(f"   Número de posições: {summary['num_positions']}")
    print(f"   Exposição: {summary['exposure_pct']:.1f}%")
    print(f"   Total de trades: {summary['num_trades']}")
    
    # ============ POSIÇÕES FINAIS ============
    if not results['positions'].empty:
        print("\n" + "="*70)
        print("📋 POSIÇÕES FINAIS (Top 10)")
        print("="*70)
        print()
        print(results['positions'].head(10).to_string(index=False))
    else:
        print("\n⚠️ Portfólio zerado (sem posições ao final)")
    
    # ============ ESTATÍSTICAS DE TRADES ============
    if not results['trades'].empty:
        print("\n" + "="*70)
        print("📈 ESTATÍSTICAS DE TRADES")
        print("="*70)
        
        trades_df = results['trades']
        
        buys = trades_df[trades_df['action'] == 'BUY']
        sells = trades_df[trades_df['action'] == 'SELL']
        
        print(f"\n   Total de trades: {len(trades_df)}")
        print(f"   Compras: {len(buys)}")
        print(f"   Vendas: {len(sells)}")
        print(f"   Comissões totais: R$ {trades_df['commission'].sum():,.2f}")
        
        # Trades por razão
        print(f"\n   Trades por razão:")
        reason_counts = trades_df['reason'].value_counts()
        for reason, count in reason_counts.items():
            print(f"      {reason}: {count}")
        
        # Top 5 ações mais negociadas
        print(f"\n   Top 5 ações mais negociadas:")
        top_traded = trades_df['ticker'].value_counts().head(5)
        for ticker, count in top_traded.items():
            print(f"      {ticker}: {count} trades")
    
    # ============ COMPARAÇÃO MENSAL COM CDI ============
    print("\n" + "="*70)
    print("📅 RETORNOS MENSAIS vs CDI")
    print("="*70)
    
    history_df = results['history']
    cdi_series = results['cdi']
    
    # Retornos mensais do portfólio
    portfolio_returns = history_df['returns'] / 100
    monthly_portfolio = portfolio_returns.resample('M').apply(
        lambda x: ((1 + x).prod() - 1) * 100
    )
    
    # Retornos mensais do CDI
    monthly_cdi = cdi_series.resample('M').apply(
        lambda x: ((1 + x).prod() - 1) * 100
    )
    
    # Combina
    monthly_comparison = pd.DataFrame({
        'Portfolio': monthly_portfolio,
        'CDI': monthly_cdi
    })
    monthly_comparison['Diferenca'] = monthly_comparison['Portfolio'] - monthly_comparison['CDI']
    
    print()
    print(monthly_comparison.to_string())
    
    # Resumo anual
    print("\n" + "="*70)
    print("📊 RESUMO POR ANO")
    print("="*70)
    
    yearly_portfolio = portfolio_returns.resample('Y').apply(
        lambda x: ((1 + x).prod() - 1) * 100
    )
    yearly_cdi = cdi_series.resample('Y').apply(
        lambda x: ((1 + x).prod() - 1) * 100
    )
    
    yearly_comparison = pd.DataFrame({
        'Portfolio': yearly_portfolio,
        'CDI': yearly_cdi
    })
    yearly_comparison['Diferenca'] = yearly_comparison['Portfolio'] - yearly_comparison['CDI']
    yearly_comparison.index = yearly_comparison.index.year
    
    print()
    print(yearly_comparison.to_string())
    
    # ============ SALVA RESULTADOS ============
    print("\n" + "="*70)
    print("💾 SALVANDO RESULTADOS")
    print("="*70)
    
    # CSV do histórico
    history_path = f"{args.output}/portfolio_history.csv"
    history_df.to_csv(history_path)
    print(f"   ✅ Histórico: {history_path}")
    
    # CSV dos trades
    trades_path = f"{args.output}/trades.csv"
    results['trades'].to_csv(trades_path, index=False)
    print(f"   ✅ Trades: {trades_path}")
    
    # CSV das posições finais
    if not results['positions'].empty:
        positions_path = f"{args.output}/final_positions.csv"
        results['positions'].to_csv(positions_path, index=False)
        print(f"   ✅ Posições finais: {positions_path}")
    
    # CSV da comparação mensal
    monthly_path = f"{args.output}/monthly_comparison.csv"
    monthly_comparison.to_csv(monthly_path)
    print(f"   ✅ Comparação mensal: {monthly_path}")
    
    # JSON das métricas
    import json
    metrics_path = f"{args.output}/metrics.json"
    with open(metrics_path, 'w') as f:
        # Converte para formato serializável
        metrics_serializable = {
            k: float(v) if isinstance(v, (int, float)) else v
            for k, v in results['metrics'].items()
        }
        json.dump(metrics_serializable, f, indent=2)
    print(f"   ✅ Métricas: {metrics_path}")
    
    # ============ GERA GRÁFICOS ============
    if not args.no_plots:
        print("\n" + "="*70)
        print("📊 GERANDO GRÁFICOS")
        print("="*70)
        
        try:
            create_performance_report(results, save_dir=args.output)
        except Exception as e:
            print(f"   ⚠️ Erro ao gerar gráficos: {e}")
            print("   Continuando sem gráficos...")
    
    # ============ RELATÓRIO FINAL ============
    print("\n" + "="*70)
    print("✅ BACKTEST CONCLUÍDO COM SUCESSO")
    print("="*70)
    
    print(f"\n🎯 RESULTADO FINAL:")
    print(f"   Retorno Total: {summary['total_return_pct']:+.2f}%")
    print(f"   Retorno Anualizado: {results['metrics']['annualized_return_pct']:+.2f}%")
    print(f"   CDI (período): {results['metrics']['cdi_total_return_pct']:.2f}%")
    print(f"   Outperformance: {results['metrics']['outperformance_pct']:+.2f}%")
    print(f"   Sharpe Ratio: {results['metrics']['sharpe_ratio']:.3f}")
    print(f"   Max Drawdown: {results['metrics']['max_drawdown_pct']:.2f}%")
    
    print(f"\n📁 Todos os arquivos salvos em: {args.output}/")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":

    main()
