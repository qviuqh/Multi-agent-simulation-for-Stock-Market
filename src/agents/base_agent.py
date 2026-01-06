import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Optional, List

class BaseAgent(ABC):
    """Base class cho tất cả agents"""
    
    def __init__(self, agent_id: str, config: Dict):
        self.agent_id = agent_id
        self.config = config
        
        # Portfolio tracking
        self.cash = config.get('initial_cash', 10000.0)
        self.inventory = config.get('initial_inventory', 0.0)
        self.portfolio_history = []
        
        # Trading history
        self.trades = []
    
    @abstractmethod
    def decide(self, observation: Dict) -> Optional[Dict]:
        """
        Quyết định hành động dựa trên observation
        
        Returns:
            {'type': 'buy'/'sell', 'size': float} or None
        """
        pass
    
    def update_portfolio(self, execution: Dict):
        """Cập nhật portfolio sau khi execute"""
        if execution['type'] == 'buy':
            self.cash -= execution['cost']
            self.inventory += execution['size']
        elif execution['type'] == 'sell':
            self.cash += execution['revenue']
            self.inventory -= execution['size']
        
        self.trades.append(execution)
    
    def get_portfolio_value(self, current_price: float) -> float:
        """Tính tổng giá trị portfolio"""
        return self.cash + self.inventory * current_price
    
    def get_pnl(self, current_price: float, initial_value: float) -> float:
        """Tính PnL"""
        return self.get_portfolio_value(current_price) - initial_value
    
    def reset(self):
        """Reset agent về trạng thái ban đầu"""
        self.cash = self.config.get('initial_cash', 10000.0)
        self.inventory = self.config.get('initial_inventory', 0.0)
        self.trades = []
        self.portfolio_history = []


class NoiseTrader(BaseAgent):
    """
    Noise/Liquidity Trader
    Giao dịch random, tạo thanh khoản nền
    """
    
    def __init__(self, agent_id: str, config: Dict):
        super().__init__(agent_id, config)
        self.trade_prob = config.get('trade_prob', 0.3)
        self.size_mean = config.get('size_mean', 10.0)
        self.size_std = config.get('size_std', 3.0)
    
    def decide(self, observation: Dict) -> Optional[Dict]:
        # Random trade với xác suất trade_prob
        if np.random.random() > self.trade_prob:
            return None
        
        # Random size (lognormal để luôn dương)
        size = np.random.lognormal(
            mean=np.log(self.size_mean), 
            sigma=self.size_std / self.size_mean
        )
        
        # Random direction
        order_type = np.random.choice(['buy', 'sell'])
        
        return {
            'type': order_type,
            'size': size,
            'agent_id': self.agent_id
        }


class ValueTrader(BaseAgent):
    """
    Value Trader
    Giao dịch dựa trên fundamental value
    """
    
    def __init__(self, agent_id: str, config: Dict):
        super().__init__(agent_id, config)
        self.fundamental_value = config.get('fundamental_value', 100.0)
        self.reversion_speed = config.get('reversion_speed', 0.01)
        self.noise_std = config.get('noise_std', 0.5)
        self.threshold_pct = config.get('threshold_pct', 0.02)  # 2%
        self.base_size = config.get('base_size', 10.0)
        
        # Fundamental drifts slowly
        self.update_fundamental()
    
    def update_fundamental(self):
        """Fundamental mean-reverting với noise"""
        target = self.config.get('fundamental_value', 100.0)
        self.fundamental_value += self.reversion_speed * (target - self.fundamental_value)
        self.fundamental_value += np.random.normal(0, self.noise_std)
    
    def decide(self, observation: Dict) -> Optional[Dict]:
        current_price = observation['mid_price']
        
        # Update fundamental value
        self.update_fundamental()
        
        # Tính deviation
        deviation = (current_price - self.fundamental_value) / self.fundamental_value
        
        # Không trade nếu trong band
        if abs(deviation) < self.threshold_pct:
            return None
        
        # Size tỉ lệ với deviation
        size = self.base_size * min(abs(deviation) / self.threshold_pct, 3.0)
        
        if deviation > 0:  # Overpriced -> sell
            return {'type': 'sell', 'size': size, 'agent_id': self.agent_id}
        else:  # Underpriced -> buy
            return {'type': 'buy', 'size': size, 'agent_id': self.agent_id}


class MomentumTrader(BaseAgent):
    """
    Momentum/Trend Follower
    Giao dịch theo trend
    """
    
    def __init__(self, agent_id: str, config: Dict):
        super().__init__(agent_id, config)
        self.short_window = config.get('short_window', 5)
        self.long_window = config.get('long_window', 20)
        self.base_size = config.get('base_size', 10.0)
        
        self.price_buffer = []
    
    def decide(self, observation: Dict) -> Optional[Dict]:
        current_price = observation['mid_price']
        self.price_buffer.append(current_price)
        
        # Chưa đủ data
        if len(self.price_buffer) < self.long_window:
            return None
        
        # Chỉ giữ window cần thiết
        if len(self.price_buffer) > self.long_window:
            self.price_buffer = self.price_buffer[-self.long_window:]
        
        # Tính MA
        ma_short = np.mean(self.price_buffer[-self.short_window:])
        ma_long = np.mean(self.price_buffer)
        
        signal = ma_short - ma_long
        
        # Threshold để tránh trade quá nhiều
        if abs(signal) < 0.01:
            return None
        
        # Size tỉ lệ với signal strength
        size = self.base_size * min(abs(signal), 3.0)
        
        if signal > 0:  # Uptrend -> buy
            return {'type': 'buy', 'size': size, 'agent_id': self.agent_id}
        else:  # Downtrend -> sell
            return {'type': 'sell', 'size': size, 'agent_id': self.agent_id}


class MeanReversionTrader(BaseAgent):
    """
    Statistical Arbitrage / Mean Reversion
    Trade khi giá xa moving average
    """
    
    def __init__(self, agent_id: str, config: Dict):
        super().__init__(agent_id, config)
        self.window = config.get('window', 20)
        self.z_threshold = config.get('z_threshold', 1.5)
        self.base_size = config.get('base_size', 10.0)
        
        self.price_buffer = []
    
    def decide(self, observation: Dict) -> Optional[Dict]:
        current_price = observation['mid_price']
        self.price_buffer.append(current_price)
        
        if len(self.price_buffer) < self.window:
            return None
        
        if len(self.price_buffer) > self.window:
            self.price_buffer = self.price_buffer[-self.window:]
        
        # Z-score
        mean_price = np.mean(self.price_buffer)
        std_price = np.std(self.price_buffer)
        
        if std_price < 1e-6:  # Tránh chia 0
            return None
        
        z_score = (current_price - mean_price) / std_price
        
        if abs(z_score) < self.z_threshold:
            return None
        
        # Size tỉ lệ với z-score
        size = self.base_size * min(abs(z_score) / self.z_threshold, 2.0)
        
        if z_score > 0:  # Price too high -> sell
            return {'type': 'sell', 'size': size, 'agent_id': self.agent_id}
        else:  # Price too low -> buy
            return {'type': 'buy', 'size': size, 'agent_id': self.agent_id}