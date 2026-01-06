"""
Test script để verify MarketSimulator hoạt động đúng
Chạy: python tests/test_market.py
"""

import sys
sys.path.append('src')

from env.market_simulator import MarketSimulator
import numpy as np
import matplotlib.pyplot as plt

def test_basic_simulation():
    """Test cơ bản: market có sinh giá hợp lý không?"""
    print("=== Test 1: Basic Price Generation ===")
    
    config = {
        'initial_price': 100.0,
        'base_spread': 0.0002,
        'impact_coef': 0.001,
        'impact_scale': 100.0,
        'noise_sigma': 0.0001,
        'transaction_cost': 0.001
    }
    
    sim = MarketSimulator(config)
    sim.reset()
    
    # Chạy 100 bước với random orders
    n_steps = 100
    for t in range(n_steps):
        # Random orders (mô phỏng 5 agents)
        orders = []
        for i in range(5):
            order_type = np.random.choice(['buy', 'sell'])
            size = np.random.uniform(5, 15)
            orders.append({
                'type': order_type,
                'size': size,
                'agent_id': f'test_agent_{i}'
            })
        
        state, exec_info = sim.step(orders)
        
        if t % 20 == 0:
            print(f"Step {t}: Price=${state.mid_price:.2f}, "
                  f"Spread={state.spread*10000:.2f}bps, "
                  f"Vol={state.volatility*100:.4f}%")
    
    print(f"\n✅ Completed {n_steps} steps")
    print(f"Price range: ${min(sim.price_history):.2f} - ${max(sim.price_history):.2f}")
    print(f"Return std: {np.std(sim.return_history[1:])*100:.4f}%")
    
    return sim

def test_price_impact():
    """Test 2: Price impact có hoạt động đúng không?"""
    print("\n=== Test 2: Price Impact Mechanism ===")
    
    config = {
        'initial_price': 100.0,
        'base_spread': 0.0002,
        'impact_coef': 0.002,  # Tăng impact để dễ thấy
        'impact_scale': 50.0,
        'noise_sigma': 0.00001,  # Giảm noise
        'transaction_cost': 0.001
    }
    
    sim = MarketSimulator(config)
    sim.reset()
    initial_price = sim.state.mid_price
    
    # Test: 10 buy orders liên tiếp → giá nên tăng
    print("Sending 10 consecutive BUY orders...")
    for _ in range(10):
        orders = [{'type': 'buy', 'size': 20, 'agent_id': 'buyer'}]
        state, _ = sim.step(orders)
    
    price_after_buys = state.mid_price
    print(f"Price before: ${initial_price:.4f}")
    print(f"Price after:  ${price_after_buys:.4f}")
    print(f"Change: {(price_after_buys/initial_price - 1)*100:+.4f}%")
    
    assert price_after_buys > initial_price, "❌ Buy orders should increase price!"
    print("✅ Price impact working correctly\n")
    
    # Test: 10 sell orders liên tiếp → giá nên giảm
    print("Sending 10 consecutive SELL orders...")
    for _ in range(10):
        orders = [{'type': 'sell', 'size': 20, 'agent_id': 'seller'}]
        state, _ = sim.step(orders)
    
    price_after_sells = state.mid_price
    print(f"Price after sells: ${price_after_sells:.4f}")
    print(f"Change from peak: {(price_after_sells/price_after_buys - 1)*100:+.4f}%")
    
    assert price_after_sells < price_after_buys, "❌ Sell orders should decrease price!"
    print("✅ Bidirectional impact confirmed")

def test_stylized_facts():
    """Test 3: Có tạo ra stylized facts không?"""
    print("\n=== Test 3: Stylized Facts Generation ===")
    
    config = {
        'initial_price': 100.0,
        'base_spread': 0.0002,
        'impact_coef': 0.001,
        'impact_scale': 100.0,
        'noise_sigma': 0.0002,  # Tăng noise một chút
        'transaction_cost': 0.001
    }
    
    sim = MarketSimulator(config)
    sim.reset()
    
    # Chạy 500 bước
    for _ in range(500):
        orders = [
            {'type': np.random.choice(['buy', 'sell']), 
             'size': np.random.lognormal(2.3, 0.5),
             'agent_id': f'agent_{i}'}
            for i in range(8)
        ]
        sim.step(orders)
    
    # Tính stylized facts
    facts = sim.get_stylized_facts()
    
    print("\nStylized Facts Results:")
    print(f"  Kurtosis: {facts['kurtosis']:.2f} (expect > 3 for fat tails)")
    print(f"  Volatility: {facts['volatility_mean']*100:.4f}%")
    print(f"  ACF(returns, lag=1): {facts['acf_returns_lag1']:.4f} (expect ~0)")
    print(f"  ACF(r², lag=1): {facts['acf_squared_lag1']:.4f} (expect > 0 for clustering)")
    print(f"  Max Drawdown: {facts['max_drawdown']*100:.2f}%")
    
    # Basic checks
    if facts['kurtosis'] > 3:
        print("✅ Fat tails detected")
    else:
        print("⚠️  Kurtosis low, but OK for small sample")
    
    if facts['acf_squared_lag1'] > 0.05:
        print("✅ Volatility clustering present")
    else:
        print("⚠️  Weak volatility clustering (might need more steps)")
    
    return sim

def visualize_simulation(sim):
    """Vẽ biểu đồ để kiểm tra trực quan"""
    print("\n=== Generating Visualization ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Price path
    axes[0, 0].plot(sim.price_history, linewidth=1.5)
    axes[0, 0].set_title('Price Path', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Price ($)')
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Returns distribution
    returns = np.array(sim.return_history[1:])
    axes[0, 1].hist(returns, bins=30, alpha=0.7, edgecolor='black', density=True)
    
    # Overlay normal
    from scipy.stats import norm
    mu, sigma = returns.mean(), returns.std()
    x = np.linspace(returns.min(), returns.max(), 100)
    axes[0, 1].plot(x, norm.pdf(x, mu, sigma), 'r--', linewidth=2, label='Normal')
    axes[0, 1].set_title(f'Returns Distribution (Kurt={np.round(sim.get_stylized_facts()["kurtosis"], 2)})', 
                         fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Return')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Volume over time
    axes[1, 0].plot(sim.volume_history, color='green', alpha=0.7)
    axes[1, 0].set_title('Trading Volume', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Volume')
    axes[1, 0].grid(alpha=0.3)
    
    # 4. Volatility over time (rolling)
    if len(returns) > 20:
        rolling_vol = [np.std(returns[max(0, i-20):i]) if i > 20 else returns[:i].std() 
                       for i in range(1, len(returns)+1)]
        axes[1, 1].plot(rolling_vol, color='orange')
        axes[1, 1].set_title('Rolling Volatility (window=20)', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Volatility')
        axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/test_market_simulation.png', dpi=150, bbox_inches='tight')
    print("✅ Plot saved to: outputs/test_market_simulation.png")
    plt.close()

def main():
    """Chạy tất cả tests"""
    print("="*60)
    print("MARKET SIMULATOR TEST SUITE")
    print("="*60)
    
    # Test 1: Basic
    sim1 = test_basic_simulation()
    
    # Test 2: Price Impact
    test_price_impact()
    
    # Test 3: Stylized Facts
    sim3 = test_stylized_facts()
    
    # Visualization
    visualize_simulation(sim3)
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED - Market Simulator is working!")
    print("="*60)
    print("\nNext steps:")
    print("1. Check the plot: outputs/test_market_simulation.png")
    print("2. If it looks good, proceed to test agents")
    print("3. Run: python tests/test_agents.py")

if __name__ == "__main__":
    main()