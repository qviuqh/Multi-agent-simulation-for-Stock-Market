"""
Experiment Runner with Unified Metrics
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Iterable
import sys
sys.path.append('src')

from env.market_simulator import MarketSimulator
from agents.base_agent import NoiseTrader, ValueTrader, MomentumTrader, MeanReversionTrader
from agents.rl_agent import RLAgent
from metrics.stylized_facts import calculate_stylized_facts, calculate_market_efficiency
from metrics.performance import calculate_agent_performance


class ExperimentRunner:
    """
    Unified experiment runner using consistent metrics
    """
    
    def __init__(self, market_config: Dict, agent_configs: Optional[Dict] = None):
        self.market_config = market_config
        self.agent_configs = agent_configs or {}
        self.results = []
    
    def run_episode(self, agents: List, n_steps: int = 1000, seed: Optional[int] = None) -> Dict:
        # sourcery skip: for-index-underscore
        """
        Run one episode with given population
        
        Returns:
            Dict containing metrics and history
        """
        if seed is not None:
            np.random.seed(seed)
        
        sim = MarketSimulator(self.market_config)
        obs = sim.reset()
        
        # Track initial values for performance calculation
        initial_values = {
            agent.agent_id: agent.get_portfolio_value(sim.state.mid_price)
            for agent in agents
        }
        
        # Episode loop
        for step in range(n_steps):
            # Collect orders from all agents
            orders = []
            for agent in agents:
                decision = agent.decide(obs)
                if decision is not None:
                    orders.append(decision)
            
            # Market step
            new_state, exec_info = sim.step(orders)
            
            # Update agent portfolios
            for execution in exec_info['executions']:
                agent_id = execution['agent_id']
                agent = next(a for a in agents if a.agent_id == agent_id)
                agent.update_portfolio(execution)
            
            # Get new observation
            obs = sim.get_observation()
        
        # Calculate metrics using unified metrics module
        metrics = self._calculate_metrics(sim, agents, initial_values)
        
        return {
            'metrics': metrics,
            'price_history': sim.price_history,
            'return_history': sim.return_history,
            'volume_history': sim.volume_history
        }
    
    def _calculate_metrics(self, sim, agents: List, initial_values: Dict) -> Dict:
        # sourcery skip: use-named-expression
        """
        Calculate all metrics using unified metrics module
        """
        returns = np.array(sim.return_history[1:])
        prices = np.array(sim.price_history)
        
        # === Market Metrics (from metrics.stylized_facts) ===
        market_facts = calculate_stylized_facts(returns, prices)
        
        # Add volume and spread
        market_facts['volume_mean'] = np.mean(sim.volume_history)
        market_facts['spread_mean'] = sim.state.spread  # Last spread
        
        # Market efficiency metrics
        fundamental_value = self.market_config.get('fundamental_value', None)
        if fundamental_value:
            efficiency_metrics = calculate_market_efficiency(prices, fundamental_value)
            market_facts.update(efficiency_metrics)
        
        # === Agent Performance (from metrics.performance) ===
        agent_metrics = {}
        current_price = sim.state.mid_price
        
        for agent in agents:
            initial_value = initial_values[agent.agent_id]
            perf = calculate_agent_performance(agent, current_price, initial_value)
            agent_metrics[agent.agent_id] = perf
        
        return {
            'market': market_facts,
            'agents': agent_metrics
        }
    
    def run_population_sweep(
        self,
        population_configs: List[Dict],
        n_seeds: int = 10,
        n_steps: int = 1000,
        seeds: Optional[Iterable[int]] = None,
    ) -> pd.DataFrame:
        """
        Experiment: Sweep over population compositions
        
        Args:
            population_configs: List of population configs like:
                {
                    'name': 'noise_heavy',
                    'noise': 10,
                    'value': 2,
                    'momentum': 2,
                    'meanrev': 1,
                    'rl': 0  # optional
                }
        """
        results = []
        seed_values = list(seeds) if seeds is not None else list(range(n_seeds))
        
        for config in population_configs:
            print(f"\nRunning {config['name']}...")
            
            for seed in seed_values:
                print(f"  Seed {seed}...", end=' ')
                
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
                
                print(f"✓ Kurt={result['metrics']['market'].get('kurtosis', 0):.2f}")
        
        return pd.DataFrame(results)
    
    def _create_agents_from_config(self, config: Dict) -> List:
        """Create agents from population config"""
        agents = []
        agent_configs = self.agent_configs
        
        # Noise traders
        for i in range(config.get('noise', 0)):
            noise_config = dict(agent_configs.get('noise', {}))
            agents.append(NoiseTrader(f'noise_{i}', noise_config))
        
        # Value traders
        for i in range(config.get('value', 0)):
            value_config = dict(agent_configs.get('value', {}))
            # Add diversity to fundamental values
            value_config['fundamental_value'] = (
                value_config.get('fundamental_value', 100.0) + 
                np.random.uniform(-2, 2)
            )
            agents.append(ValueTrader(f'value_{i}', value_config))
        
        # Momentum traders
        for i in range(config.get('momentum', 0)):
            momentum_config = dict(agent_configs.get('momentum', {}))
            agents.append(MomentumTrader(f'momentum_{i}', momentum_config))
        
        # Mean reversion traders
        for i in range(config.get('meanrev', 0)):
            meanrev_config = dict(agent_configs.get('meanrev', {}))
            agents.append(MeanReversionTrader(f'meanrev_{i}', meanrev_config))
        
        # RL traders (if specified)
        for i in range(config.get('rl', 0)):
            rl_config = dict(agent_configs.get('rl', {}))
            model_path = rl_config.pop('model_path', None)
            agents.append(RLAgent(f'rl_{i}', rl_config, model_path=model_path))
        
        return agents


# === USAGE EXAMPLE ===
if __name__ == "__main__":
    # Market config
    market_config = {
        'initial_price': 100.0,
        'base_spread': 0.0002,
        'impact_coef': 0.001,
        'noise_sigma': 0.0001,
        'transaction_cost': 0.001,
        'fundamental_value': 100.0  # For efficiency metrics
    }
    
    # Agent configs
    agent_configs = {
        'noise': {'trade_prob': 0.3, 'size_mean': 10, 'size_std': 3},
        'value': {'fundamental_value': 100, 'threshold_pct': 0.02, 'base_size': 10},
        'momentum': {'short_window': 5, 'long_window': 20, 'base_size': 10},
        'meanrev': {'window': 20, 'z_threshold': 1.5, 'base_size': 10}
    }
    
    # Population configs
    population_configs = [
        {'name': 'noise_heavy', 'noise': 15, 'value': 2, 'momentum': 2, 'meanrev': 1},
        {'name': 'balanced', 'noise': 5, 'value': 5, 'momentum': 5, 'meanrev': 5}
    ]
    
    # Run experiment
    runner = ExperimentRunner(market_config, agent_configs)
    results_df = runner.run_population_sweep(
        population_configs, 
        n_seeds=3,
        n_steps=500
    )
    
    # Display results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(results_df.groupby('config_name')[
        ['kurtosis', 'volatility_mean', 'acf_squared_lag1']
    ].mean().round(4))