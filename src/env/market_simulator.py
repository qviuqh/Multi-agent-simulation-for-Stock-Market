import numpy as np
import logging

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.stats import kurtosis
from statsmodels.tsa.stattools import acf

@dataclass
class MarketState:
    """State hiện tại của thị trường"""
    mid_price: float
    spread: float
    volume: float
    order_imbalance: float  # net buying pressure
    volatility: float
    timestamp: int
    
    def to_dict(self) -> Dict:
        return {
            'mid_price': self.mid_price,
            'spread': self.spread,
            'volume': self.volume,
            'order_imbalance': self.order_imbalance,
            'volatility': self.volatility,
            'timestamp': self.timestamp
        }

class MarketSimulator:
    """
    Level A: Price Impact + Spread Model
    Không cần full order book, tập trung vào price dynamics
    """
    
    def __init__(self, config: Dict):
        # Tham số thị trường
        self.initial_price = config.get('initial_price', 100.0)
        self.base_spread = config.get('base_spread', 0.02)  # 2 bps
        self.impact_coef = config.get('impact_coef', 0.001)
        self.impact_scale = config.get('impact_scale', 100.0)
        self.noise_sigma = config.get('noise_sigma', 0.0001)
        self.volatility_decay = config.get('volatility_decay', 0.94)
        
        # Transaction costs
        self.transaction_cost = config.get('transaction_cost', 0.001)  # 10 bps
        
        # State tracking
        self.reset()
        
        # History tracking
        self.price_history = []
        self.return_history = []
        self.volume_history = []
        
        self.logger = logging.getLogger(__name__)
    
    def reset(self):
        """Reset thị trường về trạng thái ban đầu"""
        self.state = MarketState(
            mid_price=self.initial_price,
            spread=self.base_spread,
            volume=0.0,
            order_imbalance=0.0,
            volatility=0.0001,  # Volatility ban đầu thấp
            timestamp=0
        )
        
        self.price_history = [self.initial_price]
        self.return_history = [0.0]
        self.volume_history = [0.0]
        
        return self.get_observation()
    
    def step(self, orders: List[Dict]) -> Tuple[MarketState, Dict]:
        """
        Xử lý một bước thời gian
        
        Args:
            orders: List of {'type': 'buy'/'sell', 'size': float, 'agent_id': str}
        
        Returns:
            (new_state, execution_info)
        """
        # 1. Tính net order flow
        net_flow = self._calculate_net_flow(orders)
        total_volume = sum(abs(order['size']) for order in orders)
        
        # 2. Price impact từ order flow
        impact = self._calculate_price_impact(net_flow)
        
        # 3. Noise/random walk component
        noise = np.random.normal(0, self.noise_sigma)
        
        # 4. Update mid price (log returns)
        log_return = impact + noise
        new_price = self.state.mid_price * np.exp(log_return)
        
        # 5. Update volatility estimate (EWMA)
        new_volatility = np.sqrt(
            self.volatility_decay * self.state.volatility**2 + 
            (1 - self.volatility_decay) * log_return**2
        )
        
        # 6. Update spread (phụ thuộc volatility)
        new_spread = self.base_spread * (1 + 10 * new_volatility)
        
        # 7. Execute orders và tính execution prices
        execution_info = self._execute_orders(orders, new_price, new_spread)
        
        # 8. Update state
        old_price = self.state.mid_price
        self.state = MarketState(
            mid_price=new_price,
            spread=new_spread,
            volume=total_volume,
            order_imbalance=net_flow,
            volatility=new_volatility,
            timestamp=self.state.timestamp + 1
        )
        
        # 9. Track history
        self.price_history.append(new_price)
        self.return_history.append(log_return)
        self.volume_history.append(total_volume)
        
        return self.state, execution_info
    
    def _calculate_net_flow(self, orders: List[Dict]) -> float:
        """Tính net buying pressure"""
        net = 0.0
        for order in orders:
            if order['type'] == 'buy':
                net += order['size']
            elif order['type'] == 'sell':
                net -= order['size']
        return net
    
    def _calculate_price_impact(self, net_flow: float) -> float:
        """
        Price impact function: bounded và non-linear
        Sử dụng tanh để chặn impact cực đoan
        """
        normalized_flow = net_flow / self.impact_scale
        impact = self.impact_coef * np.tanh(normalized_flow)
        return impact
    
    def _execute_orders(self, orders: List[Dict], mid_price: float, 
                       spread: float) -> Dict:
        """
        Tính execution price cho mỗi order
        Buy orders: pay ask = mid + spread/2
        Sell orders: receive bid = mid - spread/2
        """
        executions = []
        
        bid_price = mid_price - spread / 2
        ask_price = mid_price + spread / 2
        
        for order in orders:
            if order['type'] == 'buy':
                exec_price = ask_price * (1 + self.transaction_cost)
                cost = exec_price * order['size']
                executions.append({
                    'agent_id': order['agent_id'],
                    'type': 'buy',
                    'size': order['size'],
                    'price': exec_price,
                    'cost': cost
                })
            elif order['type'] == 'sell':
                exec_price = bid_price * (1 - self.transaction_cost)
                revenue = exec_price * order['size']
                executions.append({
                    'agent_id': order['agent_id'],
                    'type': 'sell',
                    'size': order['size'],
                    'price': exec_price,
                    'revenue': revenue
                })
        
        return {'executions': executions}
    
    def get_observation(self) -> Dict:
        """
        Observation cho agents (đặc biệt là RL agent)
        """
        # Recent returns (window size = 20)
        window = min(20, len(self.return_history))
        recent_returns = self.return_history[-window:] if window > 0 else [0.0]
        
        # Pad nếu chưa đủ
        if len(recent_returns) < 20:
            recent_returns = [0.0] * (20 - len(recent_returns)) + recent_returns
        
        return {
            'mid_price': self.state.mid_price,
            'spread': self.state.spread,
            'volume': self.state.volume,
            'order_imbalance': self.state.order_imbalance,
            'volatility': self.state.volatility,
            'recent_returns': np.array(recent_returns),
            'timestamp': self.state.timestamp
        }
    
    def get_stylized_facts(self) -> Dict:
        """
        Tính các stylized facts để đánh giá thị trường
        """
        returns = np.array(self.return_history[1:])  # Bỏ return đầu = 0
        
        if len(returns) < 2:
            return {}
        
        # 1. Kurtosis (fat tails)
        kurt = kurtosis(returns)
        
        # 2. Autocorrelation của returns
        acf_returns = acf(returns, nlags=10) if len(returns) > 10 else [1.0]
        
        # 3. Volatility clustering: ACF của squared returns
        acf_squared = acf(returns**2, nlags=10) if len(returns) > 10 else [1.0]
        
        # 4. Max drawdown
        prices = np.array(self.price_history)
        cummax = np.maximum.accumulate(prices)
        drawdown = (prices - cummax) / cummax
        max_dd = np.min(drawdown)
        
        return {
            'kurtosis': kurt,
            'volatility_mean': np.std(returns),
            'acf_returns_lag1': acf_returns[1] if len(acf_returns) > 1 else 0,
            'acf_squared_lag1': acf_squared[1] if len(acf_squared) > 1 else 0,
            'max_drawdown': max_dd,
            'n_steps': len(self.price_history)
        }


# === USAGE EXAMPLE ===
if __name__ == "__main__":
    # Config mẫu
    config = {
        'initial_price': 100.0,
        'base_spread': 0.0002,
        'impact_coef': 0.001,
        'impact_scale': 100.0,
        'noise_sigma': 0.0001,
        'transaction_cost': 0.0010
    }
    
    sim = MarketSimulator(config)
    sim.reset()
    
    # Mô phỏng 100 bước với random orders
    for t in range(100):
        # Random orders
        orders = [
            {'type': np.random.choice(['buy', 'sell']), 
             'size': np.random.uniform(5, 15),
             'agent_id': f'agent_{i}'}
            for i in range(5)
        ]
        
        state, exec_info = sim.step(orders)
        
        if t % 20 == 0:
            print(f"Step {t}: Price={state.mid_price:.2f}, "
                  f"Spread={state.spread:.6f}, Vol={state.volatility:.6f}")
    
    # Stylized facts
    facts = sim.get_stylized_facts()
    print("\n=== Stylized Facts ===")
    for k, v in facts.items():
        print(f"{k}: {v:.4f}")