"""
Performance metrics cho agents
"""
import numpy as np
from typing import Dict, List

def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Sharpe ratio"""
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns)


def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Sortino ratio (chỉ penalize downside volatility)"""
    if len(returns) == 0:
        return 0.0
    excess_returns = returns - risk_free_rate
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0:
        return np.inf
    downside_std = np.std(downside_returns)
    if downside_std == 0:
        return 0.0
    return np.mean(excess_returns) / downside_std


def calculate_max_drawdown(portfolio_values: np.ndarray) -> float:
    """Maximum drawdown"""
    cummax = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - cummax) / cummax
    return np.min(drawdown)


def calculate_calmar_ratio(returns: np.ndarray, portfolio_values: np.ndarray) -> float:
    """Calmar ratio = Annual return / Max drawdown"""
    if len(returns) == 0:
        return 0.0
    annual_return = np.mean(returns) * 252  # Annualize
    max_dd = abs(calculate_max_drawdown(portfolio_values))
    if max_dd == 0:
        return np.inf
    return annual_return / max_dd


def calculate_win_rate(trades: List[Dict]) -> float:
    """Win rate"""
    if len(trades) == 0:
        return 0.0
    
    winning_trades = 0
    for trade in trades:
        pnl = trade.get('revenue', 0) - trade.get('cost', 0)
        if pnl > 0:
            winning_trades += 1
    
    return winning_trades / len(trades)


def calculate_profit_factor(trades: List[Dict]) -> float:
    """Profit factor = Gross profit / Gross loss"""
    gross_profit = 0.0
    gross_loss = 0.0
    
    for trade in trades:
        pnl = trade.get('revenue', 0) - trade.get('cost', 0)
        if pnl > 0:
            gross_profit += pnl
        else:
            gross_loss += abs(pnl)
    
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0
    
    return gross_profit / gross_loss


def calculate_agent_performance(agent, current_price: float, initial_value: float) -> Dict:
    """
    Tính toán comprehensive performance metrics cho một agent
    """
    metrics = {}
    
    # Basic metrics
    final_value = agent.get_portfolio_value(current_price)
    pnl = final_value - initial_value
    metrics['pnl'] = pnl
    metrics['returns_pct'] = (pnl / initial_value) * 100
    
    # Portfolio composition
    metrics['final_cash'] = agent.cash
    metrics['final_inventory'] = agent.inventory
    metrics['inventory_value'] = agent.inventory * current_price
    
    # Trading activity
    metrics['n_trades'] = len(agent.trades)
    
    if len(agent.trades) > 0:
        # Win rate & profit factor
        metrics['win_rate'] = calculate_win_rate(agent.trades)
        metrics['profit_factor'] = calculate_profit_factor(agent.trades)
        
        # Trade returns
        trade_pnls = []
        for trade in agent.trades:
            pnl = trade.get('revenue', 0) - trade.get('cost', 0)
            trade_pnls.append(pnl / initial_value)
        
        trade_returns = np.array(trade_pnls)
        
        # Risk-adjusted metrics
        metrics['sharpe'] = calculate_sharpe_ratio(trade_returns)
        metrics['sortino'] = calculate_sortino_ratio(trade_returns)
        
        # Average trade
        metrics['avg_trade_pnl'] = np.mean(trade_pnls) * initial_value
        metrics['avg_win'] = np.mean([p for p in trade_pnls if p > 0]) * initial_value if any(p > 0 for p in trade_pnls) else 0
        metrics['avg_loss'] = np.mean([p for p in trade_pnls if p < 0]) * initial_value if any(p < 0 for p in trade_pnls) else 0
        
        # Turnover
        total_volume = sum(trade.get('size', 0) for trade in agent.trades)
        metrics['turnover'] = total_volume * current_price / initial_value
    else:
        metrics['win_rate'] = 0
        metrics['profit_factor'] = 0
        metrics['sharpe'] = 0
        metrics['sortino'] = 0
        metrics['avg_trade_pnl'] = 0
        metrics['turnover'] = 0
    
    return metrics