"""
RL Agent - Unified implementation
"""
import numpy as np
from typing import Dict, Optional
from stable_baselines3 import PPO
from .base_agent import BaseAgent


class RLAgent(BaseAgent):
    """
    RL Agent using trained PPO model
    Works as a drop-in replacement for rule-based agents
    """
    
    def __init__(self, agent_id: str, config: Dict, model_path: Optional[str] = None):
        super().__init__(agent_id, config)
        self.position_size = config.get('position_size', 10.0)
        
        # Load model if path provided
        if model_path:
            self.model = PPO.load(model_path)
            self.is_trained = True
        else:
            self.model = None
            self.is_trained = False
    
    def load_model(self, model_path: str):
        """Load a trained model"""
        self.model = PPO.load(model_path)
        self.is_trained = True
    
    def set_model(self, model: PPO):
        """Set model directly (useful for training)"""
        self.model = model
        self.is_trained = True
    
    def decide(self, observation: Dict) -> Optional[Dict]:
        """
        Use model to make trading decision
        
        Args:
            observation: Market observation from simulator
            
        Returns:
            Order dict or None
        """
        if not self.is_trained or self.model is None:
            return None
        
        # Convert observation to RL format
        obs_vector = self._observation_to_vector(observation)
        
        # Predict action
        action, _ = self.model.predict(obs_vector, deterministic=True)
        
        # Convert action to order
        return self._action_to_order(action, observation)
    
    def _observation_to_vector(self, obs: Dict) -> np.ndarray:
        # sourcery skip: inline-immediately-returned-variable
        """
        Convert market observation to RL observation format
        
        Format: [recent_returns(20), volatility, spread, volume, 
                 imbalance, inventory_norm, portfolio_return]
        """
        recent_returns = obs['recent_returns']
        volatility = obs['volatility']
        spread = obs['spread']
        volume = obs['volume']
        imbalance = obs['order_imbalance']
        
        # Portfolio features
        current_price = obs['mid_price']
        inventory_norm = self.inventory / 100.0
        initial_cash = self.config.get('initial_cash', 10000.0)
        portfolio_value = self.get_portfolio_value(current_price)
        portfolio_return = (portfolio_value - initial_cash) / initial_cash
        
        obs_vec = np.concatenate([
            recent_returns,
            [volatility, spread, volume / 100.0, imbalance / 100.0, 
             inventory_norm, portfolio_return]
        ]).astype(np.float32)
        
        return obs_vec
    
    def _action_to_order(self, action: int, obs: Dict) -> Optional[Dict]:
        """
        Convert discrete action to order
        
        Action space:
            0 = Sell
            1 = Hold
            2 = Buy
        """
        current_price = obs['mid_price']
        
        if action == 0:  # Sell
            if self.inventory >= self.position_size:
                return {
                    'type': 'sell',
                    'size': self.position_size,
                    'agent_id': self.agent_id
                }
        elif action == 2:  # Buy
            # Check if we have enough cash
            estimated_cost = current_price * self.position_size * 1.01
            if self.cash >= estimated_cost:
                return {
                    'type': 'buy',
                    'size': self.position_size,
                    'agent_id': self.agent_id
                }
        
        # action == 1 (Hold) or insufficient funds/inventory
        return None