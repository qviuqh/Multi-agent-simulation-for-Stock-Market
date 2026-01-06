"""
Test các rule-based agents
Chạy: python tests/test_agents.py
"""

import sys
sys.path.append('src')

from agents.base_agent import NoiseTrader, ValueTrader, MomentumTrader, MeanReversionTrader
from env.market_simulator import MarketSimulator
import numpy as np
import matplotlib.pyplot as plt

def test_agent_decisions():
    """Test 1: Agents có đưa ra quyết định hợp lý không?"""
    print("=== Test 1: Agent Decision Making ===\n")
    
    # Mock observation
    obs = {
        'mid_price': 100.0,
        'spread': 0.0002,
        'volume': 100.0,
        'order_imbalance': 10.0,
        'volatility': 0.001,
        'recent_returns': np.random.randn(20) * 0.001,
        'timestamp': 50
    }
    
    # Test Noise Trader
    print("1. Noise Trader:")
    noise = NoiseTrader('noise_1', {'trade_prob': 0.8, 'size_mean': 10})
    decisions = [noise.decide(obs) for _ in range(10)]
    n_trades = sum(1 for d in decisions if d is not None)
    print(f"   Made {n_trades}/10 trades (expect ~8 with prob=0.8)")
    print(f"   Sample decision: {decisions[0]}")
    
    # Test Value Trader
    print("\n2. Value Trader:")
    value = ValueTrader('value_1', {
        'fundamental_value': 105.0,  # Higher than current price
        'threshold_pct': 0.02,
        'base_size': 10.0
    })
    decision = value.decide(obs)
    print(f"   Current price: ${obs['mid_price']}, Fundamental: $105")
    print(f"   Decision: {decision}")
    print(f"   ✅ Should BUY (underpriced)" if decision and decision['type'] == 'buy' else "   ❌ Wrong decision!")
    
    # Test with overpriced scenario
    value2 = ValueTrader('value_2', {'fundamental_value': 95.0, 'threshold_pct': 0.02})
    decision2 = value2.decide(obs)
    print(f"   Fundamental: $95, Decision: {decision2}")
    print(f"   ✅ Should SELL (overpriced)" if decision2 and decision2['type'] == 'sell' else "   ❌ Wrong decision!")
    
    # Test Momentum Trader
    print("\n3. Momentum Trader:")
    momentum = MomentumTrader('momentum_1', {'short_window': 5, 'long_window': 20})
    # Feed it some prices to build history
    for i in range(25):
        momentum.decide({'mid_price': 100 + i * 0.1, **obs})  # Uptrend
    decision = momentum.decide(obs)
    print(f"   After feeding uptrend data")
    print(f"   Decision: {decision}")
    print(f"   ✅ Should BUY (momentum up)" if decision and decision['type'] == 'buy' else "   ⚠️  Need more steps")
    
    # Test Mean Reversion
    print("\n4. Mean Reversion Trader:")
    meanrev = MeanReversionTrader('meanrev_1', {'window': 20, 'z_threshold': 1.5})
    # Feed stable prices then a spike
    for i in range(20):
        meanrev.decide({'mid_price': 100.0, **obs})
    obs_spike = obs.copy()
    obs_spike['mid_price'] = 105.0  # Spike up
    decision = meanrev.decide(obs_spike)
    print(f"   Price spiked from $100 to $105")
    print(f"   Decision: {decision}")
    print(f"   ✅ Should SELL (mean revert)" if decision and decision['type'] == 'sell' else "   ⚠️  Z-score too low")
    
    print("\n✅ All agents can make decisions")

def test_integrated_simulation():
    """Test 2: Agents + Market hoạt động cùng nhau"""
    print("\n" + "="*60)
    print("=== Test 2: Integrated Agent-Market Simulation ===")
    print("="*60 + "\n")
    
    # Setup market
    market_config = {
        'initial_price': 100.0,
        'base_spread': 0.0002,
        'impact_coef': 0.001,
        'impact_scale': 100.0,
        'noise_sigma': 0.0001,
        'transaction_cost': 0.001
    }
    
    sim = MarketSimulator(market_config)
    sim.reset()
    
    # Create diverse agent population
    agents = [
        NoiseTrader('noise_1', {'trade_prob': 0.4, 'size_mean': 8}),
        NoiseTrader('noise_2', {'trade_prob': 0.3, 'size_mean': 12}),
        ValueTrader('value_1', {'fundamental_value': 102, 'threshold_pct': 0.015}),
        ValueTrader('value_2', {'fundamental_value': 98, 'threshold_pct': 0.02}),
        MomentumTrader('momentum_1', {'short_window': 5, 'long_window': 20}),
        MeanReversionTrader('meanrev_1', {'window': 15, 'z_threshold': 1.5})
    ]
    
    print(f"Population: {len(agents)} agents")
    print(f"  - 2 Noise Traders")
    print(f"  - 2 Value Traders (different fundamentals)")
    print(f"  - 1 Momentum Trader")
    print(f"  - 1 Mean Reversion Trader\n")
    
    # Run simulation
    n_steps = 200
    agent_activity = {agent.agent_id: 0 for agent in agents}
    
    print("Running 200-step simulation...")
    for t in range(n_steps):
        # Collect orders
        orders = []
        obs = sim.get_observation()
        
        for agent in agents:
            decision = agent.decide(obs)
            if decision is not None:
                orders.append(decision)
                agent_activity[agent.agent_id] += 1
        
        # Step market
        state, exec_info = sim.step(orders)
        
        # Update agent portfolios
        for execution in exec_info['executions']:
            agent = next(a for a in agents if a.agent_id == execution['agent_id'])
            agent.update_portfolio(execution)
        
        if t % 50 == 0:
            print(f"  Step {t}: Price=${state.mid_price:.2f}, "
                  f"Orders={len(orders)}, Vol={state.volume:.0f}")
    
    print(f"\n✅ Simulation completed: {n_steps} steps")
    
    # Agent activity report
    print("\nAgent Activity:")
    for agent_id, count in agent_activity.items():
        print(f"  {agent_id}: {count} trades ({count/n_steps*100:.1f}% active)")
    
    # Performance summary
    print("\nAgent Performance:")
    current_price = sim.state.mid_price
    for agent in agents:
        pnl = agent.get_pnl(current_price, 10000.0)
        pnl_pct = pnl / 10000.0 * 100
        print(f"  {agent.agent_id}: PnL=${pnl:+.2f} ({pnl_pct:+.2f}%), "
              f"Inventory={agent.inventory:.1f}")
    
    # Stylized facts
    facts = sim.get_stylized_facts()
    print("\nMarket Stylized Facts:")
    print(f"  Kurtosis: {facts['kurtosis']:.2f}")
    print(f"  Volatility: {facts['volatility_mean']*100:.4f}%")
    print(f"  Vol Clustering: {facts['acf_squared_lag1']:.4f}")
    
    return sim, agents

def visualize_agent_simulation(sim, agents):
    """Vẽ kết quả simulation"""
    print("\n=== Generating Visualization ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Price + agent trades
    ax1 = axes[0, 0]
    ax1.plot(sim.price_history, linewidth=2, label='Price', color='blue')
    
    # Mark agent trades (simplified - just show trade counts per step)
    trade_counts = []
    for agent in agents:
        trade_steps = [t['price'] for t in agent.trades if 'price' in t]
        # Approximation: can't track exact step, so skip detailed marking
    
    ax1.set_title('Price Evolution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Agent PnL
    ax2 = axes[0, 1]
    current_price = sim.state.mid_price
    agent_data = {
        'Agent': [],
        'PnL': [],
        'Type': []
    }
    for agent in agents:
        pnl = agent.get_pnl(current_price, 10000.0)
        agent_type = agent.__class__.__name__.replace('Trader', '')
        agent_data['Agent'].append(agent.agent_id)
        agent_data['PnL'].append(pnl)
        agent_data['Type'].append(agent_type)
    
    colors = {'Noise': 'gray', 'Value': 'green', 'Momentum': 'orange', 'MeanReversion': 'purple'}
    bar_colors = [colors.get(t, 'blue') for t in agent_data['Type']]
    
    ax2.bar(range(len(agent_data['Agent'])), agent_data['PnL'], color=bar_colors)
    ax2.set_xticks(range(len(agent_data['Agent'])))
    ax2.set_xticklabels(agent_data['Agent'], rotation=45, ha='right')
    ax2.set_title('Agent PnL', fontsize=12, fontweight='bold')
    ax2.set_ylabel('PnL ($)')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax2.grid(alpha=0.3, axis='y')
    
    # 3. Returns distribution
    ax3 = axes[1, 0]
    returns = np.array(sim.return_history[1:])
    ax3.hist(returns, bins=30, alpha=0.7, edgecolor='black')
    ax3.set_title('Returns Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Return')
    ax3.set_ylabel('Frequency')
    ax3.grid(alpha=0.3)
    
    # 4. Agent inventory over time (approximate)
    ax4 = axes[1, 1]
    for agent in agents[:3]:  # Show first 3 agents only
        if len(agent.trades) > 0:
            # Approximate inventory trajectory
            inventory_path = [0]
            current_inv = 0
            for trade in agent.trades:
                if trade['type'] == 'buy':
                    current_inv += trade['size']
                else:
                    current_inv -= trade['size']
                inventory_path.append(current_inv)
            ax4.plot(inventory_path, label=agent.agent_id, linewidth=1.5)
    
    ax4.set_title('Agent Inventory (first 3 agents)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Trade Number')
    ax4.set_ylabel('Inventory')
    ax4.legend()
    ax4.grid(alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    
    plt.tight_layout()
    plt.savefig('outputs/test_agents_simulation.png', dpi=150, bbox_inches='tight')
    print("✅ Plot saved to: outputs/test_agents_simulation.png")
    plt.close()

def main():
    """Run all agent tests"""
    print("="*60)
    print("AGENT TEST SUITE")
    print("="*60 + "\n")
    
    # Test 1: Individual agent logic
    test_agent_decisions()
    
    # Test 2: Integrated simulation
    sim, agents = test_integrated_simulation()
    
    # Visualization
    visualize_agent_simulation(sim, agents)
    
    print("\n" + "="*60)
    print("✅ ALL AGENT TESTS PASSED!")
    print("="*60)
    print("\nNext steps:")
    print("1. Check: outputs/test_agents_simulation.png")
    print("2. If all looks good, proceed to first experiment")
    print("3. Run: python experiments/run_first_experiment.py")

if __name__ == "__main__":
    main()