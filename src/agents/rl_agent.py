"""
RL Agent using trained PPO model
"""
import numpy as np
from typing import Dict, Optional
from stable_baselines3 import PPO
from .base_agent import BaseAgent

class RLAgent(BaseAgent):
    """
    RL Agent wrapper sử dụng trained model
    """
    
    def __init__(self, agent_id: str, config: Dict, model_path: Optional[str] = None):
        super().__init__(agent_id, config)
        self.position_size = config.get('position_size', 10.0)
        
        # Load hoặc khởi tạo model
        if model_path:
            self.model = PPO.load(model_path)
        else:
            self.model = None  # Sẽ train sau
    
    def decide(self, observation: Dict) -> Optional[Dict]:
        """Sử dụng model để quyết định"""
        if self.model is None:
            return None  # Chưa train
        
        # Convert observation sang format RL
        obs_vector = self._observation_to_vector(observation)
        
        # Predict action
        action, _ = self.model.predict(obs_vector, deterministic=True)
        
        # Convert action to order
        return self._action_to_order(action, observation)
    
    def _observation_to_vector(self, obs: Dict) -> np.ndarray:
        """Convert market observation to RL format"""
        recent_returns = obs['recent_returns']
        volatility = obs['volatility']
        spread = obs['spread']
        volume = obs['volume']
        imbalance = obs['order_imbalance']
        
        current_price = obs['mid_price']
        inventory_norm = self.inventory / 100.0
        cash_norm = (self.cash - self.config.get('initial_cash', 10000.0)) / 10000.0
        portfolio_value = self.get_portfolio_value(current_price)
        initial_cash = self.config.get('initial_cash', 10000.0)
        portfolio_return = (portfolio_value - initial_cash) / initial_cash
        
        obs_vec = np.concatenate([
            recent_returns,
            [volatility, spread, volume / 100.0, imbalance / 100.0, 
             inventory_norm, portfolio_return]
        ]).astype(np.float32)
        
        return obs_vec
    
    def _action_to_order(self, action: int, obs: Dict) -> Optional[Dict]:
        """Convert discrete action to order"""
        current_price = obs['mid_price']
        
        if action == 0:  # Sell
            if self.inventory >= self.position_size:
                return {
                    'type': 'sell',
                    'size': self.position_size,
                    'agent_id': self.agent_id
                }
        elif action == 2:  # Buy
            estimated_cost = current_price * self.position_size * 1.01
            if self.cash >= estimated_cost:
                return {
                    'type': 'buy',
                    'size': self.position_size,
                    'agent_id': self.agent_id
                }
        # action == 1 (Hold) or không đủ điều kiện
        return None