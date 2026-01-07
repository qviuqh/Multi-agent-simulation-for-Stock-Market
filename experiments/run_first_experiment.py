"""
First complete experiment: Population Composition (RQ1)
"""

import sys
sys.path.append('src')

from env.market_simulator import MarketSimulator
from agents.base_agent import NoiseTrader, ValueTrader, MomentumTrader, MeanReversionTrader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis
from statsmodels.tsa.stattools import acf
import time

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

class SimpleExperimentRunner:
    """Simplified experiment runner"""
    
    def __init__(self, market_config):
        self.market_config = market_config
    
    def create_population(self, pop_config):
        # sourcery skip: for-append-to-extend, list-comprehension
        """Create agents based on population config"""
        agents = []
        
        # Noise traders
        for i in range(pop_config.get('noise', 0)):
            agents.append(NoiseTrader(
                f'noise_{i}',
                {'trade_prob': 0.35, 'size_mean': 10, 'size_std': 3}
            ))
        
        # Value traders
        for i in range(pop_config.get('value', 0)):
            # Diverse fundamental values
            fund_val = 100 + np.random.uniform(-5, 5)
            agents.append(ValueTrader(
                f'value_{i}',
                {'fundamental_value': fund_val, 'threshold_pct': 0.02, 'base_size': 10}
            ))
        
        # Momentum traders
        for i in range(pop_config.get('momentum', 0)):
            agents.append(MomentumTrader(
                f'momentum_{i}',
                {'short_window': 5, 'long_window': 20, 'base_size': 10}
            ))
        
        # Mean reversion traders
        for i in range(pop_config.get('meanrev', 0)):
            agents.append(MeanReversionTrader(
                f'meanrev_{i}',
                {'window': 20, 'z_threshold': 1.5, 'base_size': 10}
            ))
        
        return agents
    
    def run_episode(self, agents, n_steps=500, seed=None):
        # sourcery skip: for-index-underscore
        """Run one episode"""
        if seed is not None:
            np.random.seed(seed)
        
        # Reset market
        sim = MarketSimulator(self.market_config)
        sim.reset()
        
        # Reset agents
        for agent in agents:
            agent.reset()
        
        # Initial portfolio values
        initial_values = {a.agent_id: a.get_portfolio_value(sim.state.mid_price) 
                         for a in agents}
        
        # Run simulation
        for step in range(n_steps):
            obs = sim.get_observation()
            
            # Collect orders
            orders = []
            for agent in agents:
                decision = agent.decide(obs)
                if decision is not None:
                    orders.append(decision)
            
            # Market step
            state, exec_info = sim.step(orders)
            
            # Update portfolios
            for execution in exec_info['executions']:
                agent = next(a for a in agents if a.agent_id == execution['agent_id'])
                agent.update_portfolio(execution)
        
        # Calculate metrics
        metrics = self._calculate_metrics(sim, agents, initial_values)
        
        return {
            'sim': sim,
            'agents': agents,
            'metrics': metrics
        }
    
    def _calculate_metrics(self, sim, agents, initial_values):
        """Calculate market metrics"""
        returns = np.array(sim.return_history[1:])
        prices = np.array(sim.price_history)
        
        if len(returns) < 20:
            return {}
        
        # Market metrics
        kurt = kurtosis(returns)
        vol = np.std(returns)
        acf_r = acf(returns, nlags=1)[1] if len(returns) > 10 else 0
        acf_r2 = acf(returns**2, nlags=1)[1] if len(returns) > 10 else 0
        
        cummax = np.maximum.accumulate(prices)
        drawdown = (prices - cummax) / cummax
        max_dd = np.min(drawdown)
        
        return {
            'kurtosis': kurt,
            'volatility': vol,
            'acf_return_lag1': acf_r,
            'acf_squared_lag1': acf_r2,
            'max_drawdown': max_dd,
            'spread_mean': np.mean([0.0002]),  # Simplified
            'volume_mean': np.mean(sim.volume_history),
            'price_final': prices[-1],
            'price_return_pct': (prices[-1] / prices[0] - 1) * 100
        }

def run_experiment():
    """Main experiment function"""
    print("="*70)
    print("EXPERIMENT 1: POPULATION COMPOSITION IMPACT (RQ1)")
    print("="*70)
    print("\nResearch Question:")
    print("How does the composition of trader types affect market dynamics?")
    print("\nWe will test 4 population scenarios:\n")
    
    # Market configuration
    market_config = {
        'initial_price': 100.0,
        'base_spread': 0.0002,
        'impact_coef': 0.001,
        'impact_scale': 100.0,
        'noise_sigma': 0.0002,
        'transaction_cost': 0.001
    }
    
    # Population scenarios
    populations = {
        'Noise Heavy': {'noise': 12, 'value': 2, 'momentum': 2, 'meanrev': 1},
        'Value Heavy': {'noise': 3, 'value': 10, 'momentum': 2, 'meanrev': 2},
        'Momentum Heavy': {'noise': 3, 'value': 2, 'momentum': 10, 'meanrev': 2},
        'Balanced': {'noise': 5, 'value': 5, 'momentum': 4, 'meanrev': 4}
    }
    
    for name, config in populations.items():
        total = sum(config.values())
        print(f"  {name:20s}: {total} agents "
              f"({config['noise']}N, {config['value']}V, "
              f"{config['momentum']}M, {config['meanrev']}R)")
    
    print(f"\nSettings:")
    print("  - Episode length: 500 steps")
    print("  - Seeds per scenario: 5")
    print(f"  - Total simulations: {len(populations) * 5}")
    print(f"\nEstimated time: 3-5 minutes\n")
    
    input("Press Enter to start experiment...")
    
    # Run experiments
    runner = SimpleExperimentRunner(market_config)
    results = []
    
    start_time = time.time()
    
    for pop_name, pop_config in populations.items():
        print(f"\n{'='*70}")
        print(f"Running: {pop_name}")
        print(f"{'='*70}")
        
        for seed in range(5):
            print(f"  Seed {seed+1}/5...", end=' ')
            
            # Create agents
            agents = runner.create_population(pop_config)
            
            # Run episode
            result = runner.run_episode(agents, n_steps=500, seed=seed)
            
            # Store results
            row = {
                'population': pop_name,
                'seed': seed,
                **pop_config,
                **result['metrics']
            }
            results.append(row)
            
            print(f"✓ (Price: ${result['metrics']['price_final']:.2f}, "
                  f"Kurt: {result['metrics']['kurtosis']:.2f})")
    
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"✅ Experiment completed in {elapsed:.1f} seconds")
    print(f"{'='*70}\n")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    df.to_csv('outputs/exp1_population_results.csv', index=False)
    print("📊 Results saved to: outputs/exp1_population_results.csv")
    
    return df

def analyze_results(df):
    """Analyze and visualize results"""
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70 + "\n")
    
    # Summary statistics
    print("1. SUMMARY STATISTICS BY POPULATION\n")
    summary = df.groupby('population')[
        ['kurtosis', 'volatility', 'acf_squared_lag1', 'max_drawdown']
    ].agg(['mean', 'std'])
    print(summary.round(4))
    
    # Key findings
    print("\n2. KEY FINDINGS\n")
    
    # Which population has highest kurtosis?
    kurt_by_pop = df.groupby('population')['kurtosis'].mean().sort_values(ascending=False)
    print("   Fat Tails (Kurtosis):")
    for pop, kurt in kurt_by_pop.items():
        print(f"      {pop:20s}: {kurt:.2f}")
    
    # Volatility clustering
    print(f"\n   Volatility Clustering (ACF of squared returns):")
    acf_by_pop = df.groupby('population')['acf_squared_lag1'].mean().sort_values(ascending=False)
    for pop, acf in acf_by_pop.items():
        print(f"      {pop:20s}: {acf:.4f}")
    
    # Volatility
    print(f"\n   Market Volatility:")
    vol_by_pop = df.groupby('population')['volatility'].mean().sort_values(ascending=False)
    for pop, vol in vol_by_pop.items():
        print(f"      {pop:20s}: {vol*100:.4f}%")
    
    # Create visualizations
    print("\n3. GENERATING VISUALIZATIONS...\n")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Kurtosis comparison
    ax1 = axes[0, 0]
    df.boxplot(column='kurtosis', by='population', ax=ax1)
    ax1.set_title('Fat Tails: Kurtosis by Population', fontsize=12, fontweight='bold')
    ax1.set_xlabel('')
    ax1.set_ylabel('Kurtosis')
    ax1.axhline(y=3, color='red', linestyle='--', linewidth=1, label='Normal (=3)')
    ax1.legend()
    plt.sca(ax1)
    plt.xticks(rotation=45, ha='right')
    ax1.get_figure().suptitle('')  # Remove auto title
    
    # 2. Volatility clustering
    ax2 = axes[0, 1]
    df.boxplot(column='acf_squared_lag1', by='population', ax=ax2)
    ax2.set_title('Volatility Clustering: ACF(r², lag=1)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('')
    ax2.set_ylabel('ACF')
    plt.sca(ax2)
    plt.xticks(rotation=45, ha='right')
    ax2.get_figure().suptitle('')
    
    # 3. Volatility
    ax3 = axes[1, 0]
    means = df.groupby('population')['volatility'].mean() * 100
    stds = df.groupby('population')['volatility'].std() * 100
    x_pos = np.arange(len(means))
    ax3.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='orange')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(means.index, rotation=45, ha='right')
    ax3.set_title('Market Volatility', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Volatility (%)')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Max drawdown
    ax4 = axes[1, 1]
    df.boxplot(column='max_drawdown', by='population', ax=ax4)
    ax4.set_title('Max Drawdown', fontsize=12, fontweight='bold')
    ax4.set_xlabel('')
    ax4.set_ylabel('Max Drawdown')
    plt.sca(ax4)
    plt.xticks(rotation=45, ha='right')
    ax4.get_figure().suptitle('')
    
    plt.tight_layout()
    plt.savefig('outputs/exp1_population_comparison.png', dpi=150, bbox_inches='tight')
    print("   ✅ Saved: outputs/exp1_population_comparison.png")
    
    # Statistical tests
    print("\n4. STATISTICAL SIGNIFICANCE\n")
    from scipy.stats import f_oneway
    
    groups = [df[df['population'] == pop]['kurtosis'].values 
              for pop in df['population'].unique()]
    f_stat, p_value = f_oneway(*groups)
    print("   ANOVA test for Kurtosis:")
    print(f"      F-statistic: {f_stat:.4f}")
    print(f"      p-value: {p_value:.6f}")
    if p_value < 0.05:
        print("      ✅ Populations have significantly different kurtosis (p < 0.05)")
    else:
        print("      ⚠️  No significant difference (p >= 0.05)")
    
    plt.close()

def main():
    """Main execution"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  AGENT-BASED MARKET SIMULATION - FIRST EXPERIMENT".center(68) + "║")
    print("║" + "  Population Composition Impact (RQ1)".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")
    print("\n")
    
    # Run experiment
    df = run_experiment()
    
    # Analyze results
    analyze_results(df)
    
    print("\n" + "="*70)
    print("🎉 EXPERIMENT COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  📊 outputs/exp1_population_results.csv")
    print("  📈 outputs/exp1_population_comparison.png")
    print("\nNext steps:")
    print("  1. Review the plots and CSV")
    print("  2. Write interpretation for your thesis")
    print("  3. Proceed to Experiment 2 (RL agent impact)")
    print("  4. Run: python experiments/run_rl_experiment.py")
    print("\n")

if __name__ == "__main__":
    main()