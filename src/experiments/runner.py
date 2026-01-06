import numpy as np
import pandas as pd
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis
from statsmodels.tsa.stattools import acf

class ExperimentRunner:
    """
    Chạy thí nghiệm và thu thập metrics
    """
    
    def __init__(self, market_config: Dict):
        self.market_config = market_config
        self.results = []
    
    def run_episode(self, agents: List, n_steps: int = 1000, 
                   seed: Optional[int] = None) -> Dict:
        """
        Chạy 1 episode với population cho trước
        
        Returns:
            Dict chứa metrics và history
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Import market simulator (giả sử đã có)
        from market_simulator import MarketSimulator
        
        sim = MarketSimulator(self.market_config)
        obs = sim.reset()
        
        # Initial values cho performance tracking
        initial_values = {
            agent.agent_id: agent.get_portfolio_value(sim.state.mid_price)
            for agent in agents
        }
        
        # Episode loop
        for step in range(n_steps):
            # Thu thập orders từ tất cả agents
            orders = []
            for agent in agents:
                decision = agent.decide(obs)
                if decision is not None:
                    orders.append(decision)
            
            # Market step
            new_state, exec_info = sim.step(orders)
            
            # Update agents' portfolios
            for execution in exec_info['executions']:
                agent_id = execution['agent_id']
                agent = next(a for a in agents if a.agent_id == agent_id)
                agent.update_portfolio(execution)
            
            # Get new observation
            obs = sim.get_observation()
        
        # Tính metrics
        metrics = self._calculate_metrics(sim, agents, initial_values)
        
        return {
            'metrics': metrics,
            'price_history': sim.price_history,
            'return_history': sim.return_history,
            'volume_history': sim.volume_history
        }
    
    def _calculate_metrics(self, sim, agents: List, 
                          initial_values: Dict) -> Dict:
        """Tính toán các metrics đánh giá"""
        
        returns = np.array(sim.return_history[1:])
        prices = np.array(sim.price_history)
        
        # === Market Metrics ===
        market_metrics = {
            # Fat tails
            'kurtosis': kurtosis(returns),
            
            # Volatility
            'volatility_mean': np.std(returns),
            'volatility_std': np.std([np.std(returns[i:i+20]) 
                                     for i in range(0, len(returns)-20, 20)]),
            
            # Autocorrelation
            'acf_return_lag1': acf(returns, nlags=1)[1] if len(returns) > 10 else 0,
            'acf_abs_return_lag1': acf(np.abs(returns), nlags=1)[1] if len(returns) > 10 else 0,
            'acf_squared_return_lag1': acf(returns**2, nlags=1)[1] if len(returns) > 10 else 0,
            
            # Spread & liquidity
            'spread_mean': np.mean([s.spread for s in [sim.state]]),  # Simplified
            'volume_mean': np.mean(sim.volume_history),
            
            # Crashes/bubbles
            'max_drawdown': self._max_drawdown(prices),
            'max_runup': self._max_runup(prices),
        }
        
        # === Agent Performance ===
        agent_metrics = {}
        current_price = sim.state.mid_price
        
        for agent in agents:
            final_value = agent.get_portfolio_value(current_price)
            initial_value = initial_values[agent.agent_id]
            pnl = final_value - initial_value
            returns_pct = pnl / initial_value
            
            # Sharpe (simplified)
            if len(agent.trades) > 0:
                trade_returns = [
                    (t.get('revenue', 0) - t.get('cost', 0)) / initial_value 
                    for t in agent.trades
                ]
                sharpe = np.mean(trade_returns) / (np.std(trade_returns) + 1e-8) if trade_returns else 0
            else:
                sharpe = 0
            
            agent_metrics[agent.agent_id] = {
                'pnl': pnl,
                'returns_pct': returns_pct,
                'sharpe': sharpe,
                'n_trades': len(agent.trades),
                'final_inventory': agent.inventory,
                'final_cash': agent.cash
            }
        
        return {
            'market': market_metrics,
            'agents': agent_metrics
        }
    
    def _max_drawdown(self, prices: np.ndarray) -> float:
        """Tính max drawdown"""
        cummax = np.maximum.accumulate(prices)
        drawdown = (prices - cummax) / cummax
        return np.min(drawdown)
    
    def _max_runup(self, prices: np.ndarray) -> float:
        """Tính max run-up"""
        cummin = np.minimum.accumulate(prices)
        runup = (prices - cummin) / cummin
        return np.max(runup)
    
    def run_population_sweep(self, population_configs: List[Dict], 
                            n_seeds: int = 10, n_steps: int = 1000) -> pd.DataFrame:
        """
        Thí nghiệm RQ1: Sweep population compositions
        
        Args:
            population_configs: List of dicts như:
                {
                    'name': 'noise_heavy',
                    'noise': 10,
                    'value': 2,
                    'momentum': 2,
                    'meanrev': 1
                }
        """
        results = []
        
        for config in population_configs:
            print(f"\nRunning {config['name']}...")
            
            for seed in range(n_seeds):
                # Create agents
                agents = self._create_agents_from_config(config)
                
                # Run episode
                result = self.run_episode(agents, n_steps=n_steps, seed=seed)
                
                # Store results
                row = {
                    'config_name': config['name'],
                    'seed': seed,
                    **config,  # Population counts
                    **result['metrics']['market']  # Market metrics
                }
                results.append(row)
        
        return pd.DataFrame(results)
    
    def _create_agents_from_config(self, config: Dict) -> List:
        """Helper: tạo agents từ config"""
        from base_agent import NoiseTrader, ValueTrader, MomentumTrader, MeanReversionTrader
        
        agents = []
        
        # Noise traders
        for i in range(config.get('noise', 0)):
            agents.append(NoiseTrader(
                f'noise_{i}',
                {'trade_prob': 0.3, 'size_mean': 10}
            ))
        
        # Value traders
        for i in range(config.get('value', 0)):
            agents.append(ValueTrader(
                f'value_{i}',
                {'fundamental_value': 100, 'threshold_pct': 0.02}
            ))
        
        # Momentum traders
        for i in range(config.get('momentum', 0)):
            agents.append(MomentumTrader(
                f'momentum_{i}',
                {'short_window': 5, 'long_window': 20}
            ))
        
        # Mean reversion traders
        for i in range(config.get('meanrev', 0)):
            agents.append(MeanReversionTrader(
                f'meanrev_{i}',
                {'window': 20, 'z_threshold': 1.5}
            ))
        
        return agents


# === VISUALIZATION HELPERS ===

def plot_population_comparison(df: pd.DataFrame):
    """
    Vẽ biểu đồ so sánh metrics giữa các population
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    metrics = ['kurtosis', 'volatility_mean', 'acf_squared_return_lag1',
               'max_drawdown', 'volume_mean', 'spread_mean']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        
        # Box plot
        df.boxplot(column=metric, by='config_name', ax=ax)
        ax.set_title(metric)
        ax.set_xlabel('')
        plt.sca(ax)
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig


def plot_stylized_facts(price_history: List[float], return_history: List[float]):
    """
    Vẽ stylized facts của 1 simulation
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    returns = np.array(return_history[1:])
    
    # 1. Price path
    axes[0, 0].plot(price_history)
    axes[0, 0].set_title('Price Path')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Price')
    
    # 2. Return distribution
    axes[0, 1].hist(returns, bins=50, density=True, alpha=0.7)
    # Overlay normal
    mu, sigma = returns.mean(), returns.std()
    x = np.linspace(returns.min(), returns.max(), 100)
    axes[0, 1].plot(x, 1/(sigma * np.sqrt(2 * np.pi)) * 
                   np.exp( - (x - mu)**2 / (2 * sigma**2) ),
                   'r-', label='Normal')
    axes[0, 1].set_title(f'Return Distribution (Kurt={kurtosis(returns):.2f})')
    axes[0, 1].legend()
    
    # 3. ACF of returns
    acf_vals = acf(returns, nlags=20)
    axes[1, 0].bar(range(len(acf_vals)), acf_vals)
    axes[1, 0].axhline(y=0, color='k', linestyle='--')
    axes[1, 0].set_title('ACF of Returns')
    axes[1, 0].set_xlabel('Lag')
    
    # 4. ACF of squared returns (volatility clustering)
    acf_sq = acf(returns**2, nlags=20)
    axes[1, 1].bar(range(len(acf_sq)), acf_sq)
    axes[1, 1].axhline(y=0, color='k', linestyle='--')
    axes[1, 1].set_title('ACF of Squared Returns (Vol Clustering)')
    axes[1, 1].set_xlabel('Lag')
    
    plt.tight_layout()
    return fig


# === USAGE EXAMPLE ===
if __name__ == "__main__":
    # Market config
    market_config = {
        'initial_price': 100.0,
        'base_spread': 0.0002,
        'impact_coef': 0.001,
        'noise_sigma': 0.0001,
        'transaction_cost': 0.001
    }
    
    # Population configs để sweep
    population_configs = [
        {'name': 'noise_heavy', 'noise': 15, 'value': 2, 'momentum': 2, 'meanrev': 1},
        {'name': 'value_heavy', 'noise': 5, 'value': 10, 'momentum': 2, 'meanrev': 3},
        {'name': 'momentum_heavy', 'noise': 5, 'value': 2, 'momentum': 10, 'meanrev': 3},
        {'name': 'balanced', 'noise': 5, 'value': 5, 'momentum': 5, 'meanrev': 5}
    ]
    
    # Run sweep
    runner = ExperimentRunner(market_config)
    results_df = runner.run_population_sweep(
        population_configs, 
        n_seeds=5,  # 5 seeds cho demo nhanh
        n_steps=500
    )
    
    # Analyze
    print(results_df.groupby('config_name')['kurtosis'].describe())
    
    # Plot
    fig = plot_population_comparison(results_df)
    plt.savefig('population_comparison.png', dpi=150)
    plt.show()