"""
Test trained RL agent in various scenarios

Usage:
    python scripts/test_rl_agent.py --model models/ppo_trading_agent.zip
"""
import sys
sys.path.append('src')

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from stable_baselines3 import PPO

from env.gym_wrapper import TradingEnv
from agents.base_agent import NoiseTrader, ValueTrader, MomentumTrader, MeanReversionTrader
from agents.rl_agent import RLAgent
from experiments.runner import ExperimentRunner

sns.set_style("whitegrid")


def create_test_agents(n_noise=5, n_value=3, n_momentum=3, n_meanrev=2):
    # sourcery skip: for-append-to-extend
    """Create test population"""
    agents = []
    
    for i in range(n_noise):
        agents.append(NoiseTrader(
            f'noise_{i}',
            {'trade_prob': 0.3, 'size_mean': 10.0, 'size_std': 3.0}
        ))
    
    for i in range(n_value):
        agents.append(ValueTrader(
            f'value_{i}',
            {'fundamental_value': 100.0, 'threshold_pct': 0.02, 'base_size': 10.0}
        ))
    
    for i in range(n_momentum):
        agents.append(MomentumTrader(
            f'momentum_{i}',
            {'short_window': 5, 'long_window': 20, 'base_size': 10.0}
        ))
    
    for i in range(n_meanrev):
        agents.append(MeanReversionTrader(
            f'meanrev_{i}',
            {'window': 20, 'z_threshold': 1.5, 'base_size': 10.0}
        ))
    
    return agents


def test_basic_performance(model_path: str, n_episodes: int = 10):
    """Test basic performance metrics"""
    print("="*70)
    print("TEST 1: BASIC PERFORMANCE")
    print("="*70 + "\n")
    
    # Market config
    market_config = {
        'initial_price': 100.0,
        'base_spread': 0.0002,
        'impact_coef': 0.001,
        'impact_scale': 100.0,
        'noise_sigma': 0.0002,
        'transaction_cost': 0.001
    }
    
    # Load model
    model = PPO.load(model_path)
    
    # Create test environment
    other_agents = create_test_agents()
    env = TradingEnv(market_config, other_agents, episode_length=1000)
    
    results = []
    
# sourcery skip: no-loop-in-tests
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep + 2000)
        episode_reward = 0
        n_buys = 0
        n_sells = 0
        n_holds = 0
        
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if action == 0:
                n_sells += 1
            elif action == 1:
                n_holds += 1
            elif action == 2:
                n_buys += 1
            
            done = terminated or truncated
        
        pnl = info['portfolio_value'] - env.initial_cash
        pnl_pct = (pnl / env.initial_cash) * 100
        
        results.append({
            'episode': ep,
            'reward': episode_reward,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'final_value': info['portfolio_value'],
            'final_inventory': info['inventory'],
            'n_buys': n_buys,
            'n_sells': n_sells,
            'n_holds': n_holds
        })
        
        print(f"Episode {ep+1:2d}: PnL={pnl_pct:6.2f}%, "
              f"Value=${info['portfolio_value']:8.2f}, "
              f"Actions: {n_buys}B/{n_holds}H/{n_sells}S")
    
    df = pd.DataFrame(results)
    
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    print("PnL %:")
    print(f"  Mean:   {df['pnl_pct'].mean():6.2f}%")
    print(f"  Std:    {df['pnl_pct'].std():6.2f}%")
    print(f"  Min:    {df['pnl_pct'].min():6.2f}%")
    print(f"  Max:    {df['pnl_pct'].max():6.2f}%")
    print(f"  Median: {df['pnl_pct'].median():6.2f}%")
    
    win_rate = (df['pnl'] > 0).mean() * 100
    print(f"\nWin Rate: {win_rate:.1f}% ({(df['pnl'] > 0).sum()}/{len(df)} episodes)")
    
    print(f"\nAction Distribution:")
    print(f"  Buy:  {df['n_buys'].mean():.1f} ± {df['n_buys'].std():.1f}")
    print(f"  Hold: {df['n_holds'].mean():.1f} ± {df['n_holds'].std():.1f}")
    print(f"  Sell: {df['n_sells'].mean():.1f} ± {df['n_sells'].std():.1f}")
    
    env.close()
    return df


def compare_with_baseline(model_path: str, n_episodes: int = 5):
    """Compare RL agent with baseline strategies"""
    print("\n" + "="*70)
    print("TEST 2: COMPARISON WITH BASELINES")
    print("="*70 + "\n")
    
    market_config = {
        'initial_price': 100.0,
        'base_spread': 0.0002,
        'impact_coef': 0.001,
        'impact_scale': 100.0,
        'noise_sigma': 0.0002,
        'transaction_cost': 0.001
    }
    
    agent_configs = {
        'momentum': {'short_window': 5, 'long_window': 20, 'base_size': 10.0},
        'meanrev': {'window': 20, 'z_threshold': 1.5, 'base_size': 10.0},
        'rl': {'position_size': 10.0, 'model_path': model_path}
    }
    
    # Define populations to compare
    populations = [
        {
            'name': 'Baseline (No RL)',
            'noise': 5,
            'value': 3,
            'momentum': 3,
            'meanrev': 2
        },
        {
            'name': 'With RL Agent',
            'noise': 5,
            'value': 3,
            'momentum': 3,
            'meanrev': 2,
            'rl': 1
        }
    ]
    
    runner = ExperimentRunner(market_config, agent_configs)
    results = runner.run_population_sweep(
        populations,
        n_seeds=n_episodes,
        n_steps=1000
    )
    
    # Compare market metrics
    print("Market Metrics Comparison:\n")
    comparison = results.groupby('config_name')[
        ['kurtosis', 'volatility_mean', 'acf_squared_lag1', 'max_drawdown']
    ].mean()
    
    print(comparison.round(4))
    
    # Statistical test
    from scipy.stats import ttest_ind
    
    baseline = results[results['config_name'] == 'Baseline (No RL)']['kurtosis']
    with_rl = results[results['config_name'] == 'With RL Agent']['kurtosis']
    
    t_stat, p_value = ttest_ind(baseline, with_rl)
    print(f"\nT-test for Kurtosis difference:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("  ✅ Significant difference (p < 0.05)")
    else:
        print("  ⚠️  No significant difference (p >= 0.05)")
    
    return results


def visualize_trading_behavior(model_path: str):
    # sourcery skip: remove-redundant-fstring
    """Visualize one episode of RL agent trading"""
    print("\n" + "="*70)
    print("TEST 3: TRADING BEHAVIOR VISUALIZATION")
    print("="*70 + "\n")
    
    market_config = {
        'initial_price': 100.0,
        'base_spread': 0.0002,
        'impact_coef': 0.001,
        'impact_scale': 100.0,
        'noise_sigma': 0.0002,
        'transaction_cost': 0.001
    }
    
    model = PPO.load(model_path)
    other_agents = create_test_agents()
    env = TradingEnv(market_config, other_agents, episode_length=1000)
    
    # Run one episode and record everything
    obs, _ = env.reset(seed=42)
    
    history = {
        'step': [],
        'price': [],
        'action': [],
        'inventory': [],
        'cash': [],
        'portfolio_value': []
    }
    
    done = False
    step = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        history['step'].append(step)
        history['price'].append(env.market_obs['mid_price'])
        history['action'].append(['SELL', 'HOLD', 'BUY'][action])
        history['inventory'].append(info['inventory'])
        history['cash'].append(info['cash'])
        history['portfolio_value'].append(info['portfolio_value'])
        
        step += 1
        done = terminated or truncated
    
    # Create visualization
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 1. Price and Actions
    ax1 = axes[0]
    ax1.plot(history['step'], history['price'], 'k-', alpha=0.7, linewidth=1)
    
    # Mark buy/sell actions
    for i, action in enumerate(history['action']):
        if action == 'BUY':
            ax1.scatter(i, history['price'][i], color='green', marker='^', s=50, alpha=0.7)
        elif action == 'SELL':
            ax1.scatter(i, history['price'][i], color='red', marker='v', s=50, alpha=0.7)
    
    ax1.set_ylabel('Price', fontsize=11)
    ax1.set_title('Price Path and Trading Actions', fontsize=12, fontweight='bold')
    ax1.legend(['Price', 'Buy', 'Sell'], loc='best')
    ax1.grid(alpha=0.3)
    
    # 2. Inventory
    ax2 = axes[1]
    ax2.plot(history['step'], history['inventory'], 'b-', linewidth=2)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Inventory', fontsize=11)
    ax2.set_title('Inventory Over Time', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # 3. Portfolio Value
    ax3 = axes[2]
    ax3.plot(history['step'], history['portfolio_value'], 'g-', linewidth=2)
    ax3.axhline(y=10000, color='gray', linestyle='--', alpha=0.5, label='Initial')
    ax3.set_xlabel('Time Step', fontsize=11)
    ax3.set_ylabel('Portfolio Value ($)', fontsize=11)
    ax3.set_title('Portfolio Value Over Time', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_path = 'outputs/rl_agent_behavior.png'
    Path('outputs').mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to: {output_path}")
    
    # Print summary
    final_pnl = history['portfolio_value'][-1] - 10000
    print(f"\nEpisode Summary:")
    print(f"  Initial Value: $10,000.00")
    print(f"  Final Value:   ${history['portfolio_value'][-1]:,.2f}")
    print(f"  PnL:           ${final_pnl:,.2f} ({final_pnl/10000*100:.2f}%)")
    print(f"  Final Inventory: {history['inventory'][-1]:.0f}")
    
    n_buys = history['action'].count('BUY')
    n_sells = history['action'].count('SELL')
    n_holds = history['action'].count('HOLD')
    print(f"  Actions: {n_buys} buys, {n_sells} sells, {n_holds} holds")
    
    env.close()
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Test trained RL agent")
    parser.add_argument(
        "--model",
        type=str,
        default="models/ppo_trading_agent.zip",
        help="Path to trained model"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of test episodes"
    )
    return parser.parse_args()


def main():  # sourcery skip: remove-redundant-fstring
    args = parse_args()
    
    print("\n" + "="*70)
    print("RL AGENT TESTING")
    print("="*70)
    print(f"\nModel: {args.model}")
    print(f"Test episodes: {args.episodes}\n")
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"❌ Error: Model not found at {args.model}")
        print(f"Please train a model first:")
        print(f"  python scripts/train_rl_agent.py")
        return
    
    # Run tests
    test_basic_performance(args.model, n_episodes=args.episodes)
    compare_with_baseline(args.model, n_episodes=5)
    visualize_trading_behavior(args.model)
    
    print("\n" + "="*70)
    print("🎉 ALL TESTS COMPLETED!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()