"""
Tính toán các stylized facts của thị trường
"""
import numpy as np
from scipy.stats import kurtosis, skew, jarque_bera
from statsmodels.tsa.stattools import acf, pacf
from typing import Dict, List

def calculate_stylized_facts(returns: np.ndarray, prices: np.ndarray) -> Dict:
    """
    Tính toán comprehensive stylized facts
    
    Returns:
        Dict với các metrics:
        - Fat tails: kurtosis, skewness
        - Volatility clustering: ACF of squared returns, ARCH test
        - No autocorrelation: ACF of returns
        - Long memory: Hurst exponent (optional)
    """
    
    facts = {}
    
    if len(returns) < 20:
        return facts
    
    # 1. FAT TAILS
    facts['kurtosis'] = kurtosis(returns)
    facts['excess_kurtosis'] = kurtosis(returns, fisher=True)  # -3
    facts['skewness'] = skew(returns)
    
    # Jarque-Bera test for normality
    jb_stat, jb_pvalue = jarque_bera(returns)
    facts['jarque_bera_stat'] = jb_stat
    facts['jarque_bera_pvalue'] = jb_pvalue
    facts['is_normal'] = jb_pvalue > 0.05
    
    # 2. VOLATILITY CLUSTERING
    # ACF of squared returns
    acf_squared = acf(returns**2, nlags=min(20, len(returns)//4))
    facts['acf_squared_lag1'] = acf_squared[1] if len(acf_squared) > 1 else 0
    facts['acf_squared_lag5'] = acf_squared[5] if len(acf_squared) > 5 else 0
    facts['acf_squared_mean_10'] = np.mean(acf_squared[1:11]) if len(acf_squared) > 10 else 0
    
    # ACF of absolute returns
    acf_abs = acf(np.abs(returns), nlags=min(20, len(returns)//4))
    facts['acf_abs_lag1'] = acf_abs[1] if len(acf_abs) > 1 else 0
    
    # 3. NO AUTOCORRELATION IN RETURNS
    acf_returns = acf(returns, nlags=min(20, len(returns)//4))
    facts['acf_return_lag1'] = acf_returns[1] if len(acf_returns) > 1 else 0
    facts['acf_return_mean_10'] = np.mean(np.abs(acf_returns[1:11])) if len(acf_returns) > 10 else 0
    
    # 4. DRAWDOWNS & RUNUPS
    cummax = np.maximum.accumulate(prices)
    drawdown = (prices - cummax) / cummax
    facts['max_drawdown'] = np.min(drawdown)
    facts['avg_drawdown'] = np.mean(drawdown[drawdown < 0]) if np.any(drawdown < 0) else 0
    
    cummin = np.minimum.accumulate(prices)
    runup = (prices - cummin) / cummin
    facts['max_runup'] = np.max(runup)
    
    # 5. VOLATILITY STATISTICS
    facts['volatility_mean'] = np.std(returns)
    
    # Rolling volatility để đo clustering
    window = min(20, len(returns)//5)
    if len(returns) > window:
        rolling_vols = [np.std(returns[i:i+window]) 
                       for i in range(len(returns) - window)]
        facts['volatility_of_volatility'] = np.std(rolling_vols)
    
    return facts


def calculate_market_efficiency(prices: np.ndarray, fundamental: float = None) -> Dict:
    """
    Đo lường market efficiency
    """
    metrics = {}
    
    returns = np.diff(np.log(prices))
    
    # 1. Variance Ratio Test
    # VR(q) = Var(r_t(q)) / q * Var(r_t)
    # If random walk, VR should be 1
    if len(returns) > 10:
        q_values = [2, 5, 10]
        for q in q_values:
            if len(returns) > q:
                q_returns = np.array([np.sum(returns[i:i+q]) for i in range(len(returns) - q)])
                vr = np.var(q_returns) / (q * np.var(returns))
                metrics[f'variance_ratio_q{q}'] = vr
    
    # 2. Price deviation from fundamental (nếu có)
    if fundamental is not None:
        deviations = (prices - fundamental) / fundamental
        metrics['mean_abs_deviation'] = np.mean(np.abs(deviations))
        metrics['max_deviation'] = np.max(np.abs(deviations))
    
    return metrics


def test_stylized_facts():
    """Test function"""
    # Generate sample data with known properties
    np.random.seed(42)
    
    # Fat-tailed returns
    returns = np.random.standard_t(df=5, size=1000) * 0.01
    prices = 100 * np.exp(np.cumsum(returns))
    
    facts = calculate_stylized_facts(returns, prices)
    
    print("=== Stylized Facts Test ===")
    print(f"Kurtosis: {facts['kurtosis']:.2f} (expect > 3)")
    print(f"Volatility Clustering: {facts['acf_squared_lag1']:.4f} (expect > 0)")
    print(f"ACF Returns: {facts['acf_return_lag1']:.4f} (expect ~0)")
    print(f"Max Drawdown: {facts['max_drawdown']*100:.2f}%")
    
    assert facts['kurtosis'] > 3, "Should have fat tails"
    print("✅ Test passed!")

if __name__ == "__main__":
    test_stylized_facts()