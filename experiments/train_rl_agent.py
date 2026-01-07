"""
Complete RL Agent Training Script
Train PPO agent to trade in multi-agent market

Usage:
    python scripts/train_rl_agent.py --timesteps 100000 --save-path models/ppo_agent.zip
"""
import sys
sys.path.append('src')

import argparse
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from env.gym_wrapper import TradingEnv
from agents.base_agent import NoiseTrader, ValueTrader, MomentumTrader, MeanReversionTrader


def create_other_agents(config: dict) -> list:
    # sourcery skip: for-append-to-extend
    """
    Create population of other agents for training environment
    """
    agents = []
    
    # Noise traders (provide liquidity)
    for i in range(config.get('n_noise', 5)):
        agents.append(NoiseTrader(
            f'noise_{i}',
            {
                'trade_prob': 0.3,
                'size_mean': 10.0,
                'size_std': 3.0,
                'initial_cash': 10000.0
            }
        ))
    
    # Value traders (fundamental)
    for i in range(config.get('n_value', 3)):
        fund_val = 100.0 + np.random.uniform(-3, 3)
        agents.append(ValueTrader(
            f'value_{i}',
            {
                'fundamental_value': fund_val,
                'threshold_pct': 0.02,
                'base_size': 10.0,
                'initial_cash': 10000.0
            }
        ))
    
    # Momentum traders (technical)
    for i in range(config.get('n_momentum', 3)):
        agents.append(MomentumTrader(
            f'momentum_{i}',
            {
                'short_window': 5,
                'long_window': 20,
                'base_size': 10.0,
                'initial_cash': 10000.0
            }
        ))
    
    # Mean reversion traders
    for i in range(config.get('n_meanrev', 2)):
        agents.append(MeanReversionTrader(
            f'meanrev_{i}',
            {
                'window': 20,
                'z_threshold': 1.5,
                'base_size': 10.0,
                'initial_cash': 10000.0
            }
        ))
    
    return agents


def make_env(market_config: dict, other_agents: list, episode_length: int):
    """Create and wrap environment"""
    def _init():
        env = TradingEnv(market_config, other_agents, episode_length)
        env = Monitor(env)  # Wrap with Monitor for logging
        return env
    return _init


def train_rl_agent(
    total_timesteps: int = 100000,
    save_path: str = "models/ppo_trading_agent.zip",
    eval_freq: int = 5000,
    n_eval_episodes: int = 5,
    log_dir: str = "./logs/",
    checkpoint_freq: int = 10000
):
    """
    Train PPO agent
    
    Args:
        total_timesteps: Total training steps
        save_path: Where to save final model
        eval_freq: Evaluate every N steps
        n_eval_episodes: Episodes for evaluation
        log_dir: TensorBoard log directory
        checkpoint_freq: Save checkpoint every N steps
    """
    print("="*70)
    print("RL AGENT TRAINING")
    print("="*70)
    
    # Create directories
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)
    
    # Market configuration
    market_config = {
        'initial_price': 100.0,
        'base_spread': 0.0002,
        'impact_coef': 0.001,
        'impact_scale': 100.0,
        'noise_sigma': 0.0002,
        'volatility_decay': 0.94,
        'transaction_cost': 0.001
    }
    
    # Create other agents
    agent_config = {
        'n_noise': 5,
        'n_value': 3,
        'n_momentum': 3,
        'n_meanrev': 2
    }
    other_agents = create_other_agents(agent_config)
    
    print(f"\nEnvironment Setup:")
    print(f"  Total agents: {len(other_agents)} + 1 RL agent")
    print("  Episode length: 1000 steps")
    print(f"  Transaction cost: {market_config['transaction_cost']*10000:.1f} bps")
    
    # Create training environment
    episode_length = 1000
    env = DummyVecEnv([make_env(market_config, other_agents, episode_length)])
    
    # Create evaluation environment (separate agents)
    eval_agents = create_other_agents(agent_config)
    eval_env = DummyVecEnv([make_env(market_config, eval_agents, episode_length)])
    
    print(f"\nTraining Setup:")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Eval frequency: {eval_freq:,} steps")
    print(f"  Checkpoint frequency: {checkpoint_freq:,} steps")
    
    # Create callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(Path(save_path).parent / "best_model"),
        log_path=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path="./checkpoints/",
        name_prefix="ppo_trading",
        verbose=1
    )
    
    # Create PPO model
    print(f"\nCreating PPO model...")
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
        ent_coef=0.01,  # Encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=log_dir
    )
    
    print(f"\nModel Architecture:")
    print("  Policy: MLP (Multi-Layer Perceptron)")
    print(f"  Observation space: {env.observation_space.shape}")
    print("  Action space: Discrete(3) [Sell, Hold, Buy]")
    print("  Learning rate: 3e-4")
    
    # Train
    print(f"\n{'='*70}")
    print("TRAINING STARTED")
    print(f"{'='*70}\n")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True
        )
        
        print(f"\n{'='*70}")
        print("TRAINING COMPLETED")
        print(f"{'='*70}")
        
        # Save final model
        model.save(save_path)
        print(f"\n✅ Model saved to: {save_path}")
        
        # Test the trained model
        print(f"\n{'='*70}")
        print("TESTING TRAINED MODEL")
        print(f"{'='*70}\n")
        
        test_trained_model(model, market_config, agent_config)
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Training interrupted by user")
        print(f"Saving current model to: {save_path}")
        model.save(save_path)
        print("✅ Model saved")
    
    env.close()
    eval_env.close()
    
    return model


def test_trained_model(model, market_config: dict, agent_config: dict, n_episodes: int = 3):
    """
    Test trained model on fresh episodes
    """
    # Create test environment
    test_agents = create_other_agents(agent_config)
    test_env = TradingEnv(market_config, test_agents, episode_length=1000)
    
    episode_rewards = []
    episode_pnls = []
    
    for ep in range(n_episodes):
        obs, _ = test_env.reset(seed=ep + 1000)
        episode_reward = 0
        initial_value = test_env.initial_cash
        
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        final_value = info['portfolio_value']
        pnl = final_value - initial_value
        pnl_pct = (pnl / initial_value) * 100
        
        episode_rewards.append(episode_reward)
        episode_pnls.append(pnl_pct)
        
        print(f"  Episode {ep+1}: Reward={episode_reward:.2f}, "
              f"PnL={pnl_pct:.2f}%, Final Value=${final_value:.2f}")
    
    print(f"\nTest Summary:")
    print(f"  Avg Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Avg PnL: {np.mean(episode_pnls):.2f}% ± {np.std(episode_pnls):.2f}%")
    
    test_env.close()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train RL trading agent")
    
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Total training timesteps (default: 100000)"
    )
    
    parser.add_argument(
        "--save-path",
        type=str,
        default="outputs/models/ppo_trading_agent.zip",
        help="Path to save trained model"
    )
    
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=5000,
        help="Evaluate model every N steps"
    )
    
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=10000,
        help="Save checkpoint every N steps"
    )
    
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs/",
        help="TensorBoard log directory"
    )
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    print("\n" + "="*70)
    print("AGENT-BASED MARKET SIMULATION - RL TRAINING")
    print("="*70 + "\n")
    
    model = train_rl_agent(
        total_timesteps=args.timesteps,
        save_path=args.save_path,
        eval_freq=args.eval_freq,
        checkpoint_freq=args.checkpoint_freq,
        log_dir=args.log_dir
    )
    
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE!")
    print("="*70)
    print(f"\nGenerated files:")
    print(f"  🤖 Final model: {args.save_path}")
    print(f"  🏆 Best model: {Path(args.save_path).parent / 'best_model'}")
    print(f"  📊 TensorBoard logs: {args.log_dir}")
    print("  💾 Checkpoints: ./checkpoints/")
    
    print(f"\nNext steps:")
    print("  1. View training progress:")
    print(f"     tensorboard --logdir {args.log_dir}")
    print("  2. Test the model:")
    print(f"     python scripts/test_rl_agent.py --model {args.save_path}")
    print("  3. Run experiments with RL agent:")
    print("     python experiments/run_rl_experiment.py")
    print()


if __name__ == "__main__":
    main()