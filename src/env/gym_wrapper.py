import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from ..agents.base_agent import NoiseTrader, ValueTrader, MomentumTrader, MeanReversionTrader
from market_simulator import MarketSimulator

class TradingEnv(gym.Env):
    """
    Gymnasium environment cho RL agent
    Observation: market state + agent portfolio
    Action: {Sell, Hold, Buy} hoặc continuous position target
    """
    
    def __init__(self, market_config: Dict, other_agents: List, 
                 episode_length: int = 1000):
        super().__init__()
        
        self.market_config = market_config
        self.other_agents = other_agents
        self.episode_length = episode_length
        
        # Action space: Discrete 3 actions
        # 0 = Sell (target position = -1)
        # 1 = Hold (target position = 0)
        # 2 = Buy (target position = +1)
        self.action_space = spaces.Discrete(3)
        
        # Observation space
        # [recent_returns(20), volatility, spread, volume, imbalance, 
        #  inventory, cash_norm, portfolio_return]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(20 + 6,),  # 20 returns + 6 features
            dtype=np.float32
        )
        
        # Agent state
        self.initial_cash = 10000.0
        self.cash = self.initial_cash
        self.inventory = 0.0
        self.position_size = 10.0  # Mỗi lệnh mua/bán 10 shares
        
        # Market simulator (sẽ reset trong reset())
        self.sim = None
        self.step_count = 0
    
    def reset(self, seed=None, options=None):
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
        """
        Execute one step
        
        Action mapping:
            0 -> Sell (nếu có inventory)
            1 -> Hold
            2 -> Buy (nếu có cash)
        """
        # 1. Convert action to order
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
            'cash': self.cash
        }
        
        return obs, reward, terminated, truncated, info
    
    def _action_to_order(self, action: int) -> Dict:
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
            # Check if we have enough cash (rough estimate)
            estimated_cost = current_price * self.position_size * 1.01  # +1% buffer
            if self.cash >= estimated_cost:
                return {
                    'type': 'buy',
                    'size': self.position_size,
                    'agent_id': 'rl_agent'
                }
        # action == 1 (Hold) or không đủ điều kiện -> return None
        return None
    
    def _update_portfolio(self, execution: Dict):
        """Update RL agent's portfolio"""
        if execution['type'] == 'buy':
            self.cash -= execution['cost']
            self.inventory += execution['size']
        elif execution['type'] == 'sell':
            self.cash += execution['revenue']
            self.inventory -= execution['size']
    
    def _get_observation(self) -> np.ndarray:
        """
        Construct observation vector
        """
        # Market features
        recent_returns = self.market_obs['recent_returns']  # (20,)
        volatility = self.market_obs['volatility']
        spread = self.market_obs['spread']
        volume = self.market_obs['volume']
        imbalance = self.market_obs['order_imbalance']
        
        # Portfolio features
        current_price = self.market_obs['mid_price']
        inventory_norm = self.inventory / 100.0  # Normalize
        cash_norm = (self.cash - self.initial_cash) / self.initial_cash
        portfolio_value = self.get_portfolio_value(current_price)
        portfolio_return = (portfolio_value - self.initial_cash) / self.initial_cash
        
        obs = np.concatenate([
            recent_returns,
            [volatility, spread, volume / 100.0, imbalance / 100.0,
             inventory_norm, portfolio_return]
        ]).astype(np.float32)
        
        return obs
    
    def _calculate_reward(self, action: int, new_state) -> float:
        """
        Reward function
        
        Components:
        1. Portfolio value change (PnL)
        2. Inventory risk penalty
        3. Transaction cost penalty
        """
        current_value = self.get_portfolio_value(new_state.mid_price)
        
        # PnL component
        pnl = current_value - self.initial_cash
        pnl_reward = pnl / self.initial_cash * 100  # Scale up
        
        # Inventory risk penalty (penalize large positions)
        inventory_penalty = -0.01 * (self.inventory / 100.0) ** 2
        
        # Transaction cost (implicitly in PnL, but can add explicit penalty)
        transaction_penalty = 0
        if action != 1:  # Nếu không phải Hold
            transaction_penalty = -0.001
        
        reward = pnl_reward + inventory_penalty + transaction_penalty
        
        return reward
    
    def get_portfolio_value(self, current_price: float) -> float:
        """Calculate total portfolio value"""
        return self.cash + self.inventory * current_price


# === RL AGENT WRAPPER ===

class RLAgentWrapper:
    """
    Wrapper để RL agent tương tác như rule-based agent
    """
    
    def __init__(self, model: PPO, agent_id: str = 'rl_agent'):
        self.model = model
        self.agent_id = agent_id
        self.cash = 10000.0
        self.inventory = 0.0
        self.trades = []
        
        # Internal state để xây dựng observation
        self.position_size = 10.0
    
    def decide(self, observation: Dict) -> Dict:
        """
        Sử dụng model để quyết định action
        """
        # Convert market observation to RL observation format
        obs = self._market_obs_to_rl_obs(observation)
        
        # Predict action
        action, _ = self.model.predict(obs, deterministic=True)
        
        # Convert action to order
        return self._action_to_order(action, observation)
    
    def _market_obs_to_rl_obs(self, market_obs: Dict) -> np.ndarray:
        """Convert market observation to RL format"""
        recent_returns = market_obs['recent_returns']
        volatility = market_obs['volatility']
        spread = market_obs['spread']
        volume = market_obs['volume']
        imbalance = market_obs['order_imbalance']
        
        current_price = market_obs['mid_price']
        inventory_norm = self.inventory / 100.0
        cash_norm = (self.cash - 10000.0) / 10000.0
        portfolio_value = self.cash + self.inventory * current_price
        portfolio_return = (portfolio_value - 10000.0) / 10000.0
        
        obs = np.concatenate([
            recent_returns,
            [volatility, spread, volume / 100.0, imbalance / 100.0, inventory_norm, portfolio_return]
        ]).astype(np.float32)
        
        return obs
    
    def _action_to_order(self, action: int, market_obs: Dict) -> Dict:
        """Same logic as TradingEnv"""
        current_price = market_obs['mid_price']
        
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
        return None
    
    def update_portfolio(self, execution: Dict):
        """Update portfolio after execution"""
        if execution['type'] == 'buy':
            self.cash -= execution['cost']
            self.inventory += execution['size']
        elif execution['type'] == 'sell':
            self.cash += execution['revenue']
            self.inventory -= execution['size']
        
        self.trades.append(execution)
    
    def get_portfolio_value(self, current_price: float) -> float:
        return self.cash + self.inventory * current_price
    
    def reset(self):
        self.cash = 10000.0
        self.inventory = 0.0
        self.trades = []


# === TRAINING SCRIPT ===

def train_rl_agent(market_config: Dict, other_agents: List, 
                   total_timesteps: int = 100000):
    """
    Train PPO agent
    """
    # Create environment
    env = TradingEnv(market_config, other_agents, episode_length=1000)
    env = DummyVecEnv([lambda: env])
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./ppo_trading_tensorboard/"
    )
    
    # Train
    print("Training RL agent...")
    model.learn(total_timesteps=total_timesteps)
    
    # Save
    model.save("ppo_trading_agent")
    print("Model saved!")
    
    return model


# === USAGE EXAMPLE ===
if __name__ == "__main__":
    
    # Market config
    market_config = {
        'initial_price': 100.0,
        'base_spread': 0.0002,
        'impact_coef': 0.001,
        'noise_sigma': 0.0001,
        'transaction_cost': 0.001
    }
    
    # Other agents
    other_agents = [
        NoiseTrader('noise_1', {'trade_prob': 0.3, 'size_mean': 10}),
        NoiseTrader('noise_2', {'trade_prob': 0.3, 'size_mean': 10}),
        ValueTrader('value_1', {'fundamental_value': 100, 'threshold_pct': 0.02}),
        MomentumTrader('momentum_1', {'short_window': 5, 'long_window': 20}),
        MeanReversionTrader('meanrev_1', {'window': 20, 'z_threshold': 1.5})
    ]
    
    # Train
    model = train_rl_agent(market_config, other_agents, total_timesteps=50000)
    
    # Test trained agent
    print("\nTesting trained agent...")
    env = TradingEnv(market_config, other_agents)
    obs, _ = env.reset()
    
    total_reward = 0
    for _ in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    print(f"Test episode - Total reward: {total_reward:.2f}")
    print(f"Final portfolio value: {info['portfolio_value']:.2f}")