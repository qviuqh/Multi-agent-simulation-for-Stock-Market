"""
Gymnasium environment for RL agent training
"""
import sys
sys.path.append('src')

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional
from env.market_simulator import MarketSimulator


class TradingEnv(gym.Env):
    """
    Gymnasium environment for RL agent training
    
    Observation: [recent_returns(20), volatility, spread, volume, imbalance, inventory_norm, portfolio_return]
    Action: Discrete(3) - {Sell=0, Hold=1, Buy=2}
    """
    
    def __init__(self, market_config: Dict, other_agents: List, 
                 episode_length: int = 1000):
        super().__init__()
        
        self.market_config = market_config
        self.other_agents = other_agents
        self.episode_length = episode_length
        
        # Action space: 3 discrete actions
        self.action_space = spaces.Discrete(3)
        
        # Observation space: 20 returns + 6 features
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(26,),
            dtype=np.float32
        )
        
        # Agent parameters
        self.initial_cash = 10000.0
        self.position_size = 10.0
        
        # State (will be initialized in reset())
        self.sim = None
        self.cash = self.initial_cash
        self.inventory = 0.0
        self.step_count = 0
        self.market_obs = None
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Reset market
        self.sim = MarketSimulator(self.market_config)
        self.market_obs = self.sim.reset()
        
        # Reset agent state
        self.cash = self.initial_cash
        self.inventory = 0.0
        self.step_count = 0
        
        # Reset other agents
        for agent in self.other_agents:
            agent.reset()
        
        obs = self._get_observation()
        info = {}
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        
        # 1. Convert RL action to order
        rl_order = self._action_to_order(action)
        
        # 2. Get orders from other agents
        other_orders = []
        for agent in self.other_agents:
            decision = agent.decide(self.market_obs)
            if decision is not None:
                other_orders.append(decision)
        
        # 3. Combine all orders
        all_orders = other_orders
        if rl_order is not None:
            all_orders.append(rl_order)
        
        # 4. Market step
        new_state, exec_info = self.sim.step(all_orders)
        
        # 5. Update portfolios
        for execution in exec_info['executions']:
            if execution['agent_id'] == 'rl_agent':
                self._update_portfolio(execution)
            else:
                # Update other agents
                agent = next(a for a in self.other_agents if a.agent_id == execution['agent_id'])
                agent.update_portfolio(execution)
        
        # 6. Get new observation
        self.market_obs = self.sim.get_observation()
        obs = self._get_observation()
        
        # 7. Calculate reward
        reward = self._calculate_reward(action, new_state)
        
        # 8. Check termination
        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.episode_length
        
        info = {
            'portfolio_value': self.get_portfolio_value(new_state.mid_price),
            'inventory': self.inventory,
            'cash': self.cash,
            'step': self.step_count
        }
        
        return obs, reward, terminated, truncated, info
    
    def _action_to_order(self, action: int) -> Optional[Dict]:
        """Convert discrete action to order"""
        current_price = self.market_obs['mid_price']
        
        if action == 0:  # Sell
            if self.inventory >= self.position_size:
                return {
                    'type': 'sell',
                    'size': self.position_size,
                    'agent_id': 'rl_agent'
                }
        elif action == 2:  # Buy
            estimated_cost = current_price * self.position_size * 1.01
            if self.cash >= estimated_cost:
                return {
                    'type': 'buy',
                    'size': self.position_size,
                    'agent_id': 'rl_agent'
                }
        
        return None
    
    def _update_portfolio(self, execution: Dict):
        """Update RL agent's portfolio after execution"""
        if execution['type'] == 'buy':
            self.cash -= execution['cost']
            self.inventory += execution['size']
        elif execution['type'] == 'sell':
            self.cash += execution['revenue']
            self.inventory -= execution['size']
    
    def _get_observation(self) -> np.ndarray:
        # sourcery skip: inline-immediately-returned-variable
        """
        Construct observation vector
        Same format as RLAgent._observation_to_vector()
        """
        recent_returns = self.market_obs['recent_returns']
        volatility = self.market_obs['volatility']
        spread = self.market_obs['spread']
        volume = self.market_obs['volume']
        imbalance = self.market_obs['order_imbalance']
        
        # Portfolio features
        current_price = self.market_obs['mid_price']
        inventory_norm = self.inventory / 100.0
        portfolio_value = self.get_portfolio_value(current_price)
        portfolio_return = (portfolio_value - self.initial_cash) / self.initial_cash
        
        obs = np.concatenate([
            recent_returns,
            [volatility, spread, volume / 100.0, imbalance / 100.0,
             inventory_norm, portfolio_return]
        ]).astype(np.float32)
        
        return obs
    
    def _calculate_reward(self, action: int, new_state) -> float:
        # sourcery skip: inline-immediately-returned-variable
        """
        Reward function
        
        Components:
        1. PnL change (main signal)
        2. Inventory risk penalty
        3. Transaction cost penalty
        """
        current_value = self.get_portfolio_value(new_state.mid_price)
        
        # PnL component (scaled)
        pnl = current_value - self.initial_cash
        pnl_reward = pnl / self.initial_cash * 100
        
        # Inventory risk penalty (discourage large positions)
        inventory_penalty = -0.01 * (self.inventory / 100.0) ** 2
        
        # Transaction cost penalty
        transaction_penalty = -0.001 if action != 1 else 0
        
        reward = pnl_reward + inventory_penalty + transaction_penalty
        
        return reward
    
    def get_portfolio_value(self, current_price: float) -> float:
        """Calculate total portfolio value"""
        return self.cash + self.inventory * current_price