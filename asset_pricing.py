import numpy as np

def simulate_economy(
    T=10_000,
    beta=0.99,
    gamma=2.0,
    alpha=0.0,            # habit strength: 0 = none, 0.8 = strong
    mu_c=0.001,           # average consumption growth
    sigma_c=0.02,         # volatility of consumption growth
    mu_e=0.005,           # risky asset excess return drift
    sigma_e=0.10,         # risky asset volatility
    rho_ce=0.6,           # correlation between cons. and risky asset shocks
    seed=0
):
    """
    Simulate a simple habit-formation economy and compute:
    - average risk-free rate
    - average risky asset return
    - equity premium
    """
    rng = np.random.default_rng(seed)

    # --- 1. Simulate shocks for consumption and risky asset ---
    # Draw correlated shocks for consumption and risky asset
    # We build 2D correlated normals via Cholesky.
    cov = np.array([[sigma_c**2, rho_ce * sigma_c * sigma_e],
                    [rho_ce * sigma_c * sigma_e, sigma_e**2]])
    mean = np.array([mu_c, mu_e])
    shocks = rng.multivariate_normal(mean, cov, size=T)

    gc = shocks[:, 0]  # consumption growth (log approx)
    re_shock = shocks[:, 1]  # risky asset log-return shock (excess)

    # --- 2. Build consumption path ---
    C = np.zeros(T + 1)
    C[0] = 1.0  # normalize initial consumption

    for t in range(T):
        # log C_{t+1} = log C_t + gc_t  (approx)
        C[t + 1] = C[t] * np.exp(gc[t])

    # --- 3. Surplus consumption with habit ---
    surplus = np.zeros(T + 1)
    surplus[0] = C[0]  # no habit at t=0

    for t in range(1, T + 1):
        surplus[t] = C[t] - alpha * C[t - 1]
        # avoid negative / zero surplus (numerical safety)
        surplus[t] = max(surplus[t], 1e-6)

    # --- 4. Marginal utility and SDF ---
    m = surplus ** (-gamma)           # marginal utility
    M = beta * (m[1:] / m[:-1])       # SDF M_{t+1}

    # --- 5. Risk-free rate and risky asset returns ---
    # Risk-free: 1/Rf ≈ E[M]  => Rf ≈ 1 / E[M]
    Rf = 1.0 / np.mean(M)

    # Risky asset: assume log return = re_shock, so:
    Re = np.exp(re_shock)            # gross return on equity
    # Price condition in theory: E[M * Re] = 1
    # Here we just look at average realized returns and premium
    mean_Re = np.mean(Re)
    equity_premium = mean_Re - Rf

    return {
        "alpha": alpha,
        "Rf_mean": Rf,
        "Re_mean": mean_Re,
        "equity_premium": equity_premium
    }


if __name__ == "__main__":
    # No habit
    res_no_habit = simulate_economy(alpha=0.0)
    # Strong habit
    res_habit = simulate_economy(alpha=0.8)

    print("=== No habit (alpha = 0.0) ===")
    print(f"Average risk-free rate      : {res_no_habit['Rf_mean']:.4f}")
    print(f"Average risky asset return  : {res_no_habit['Re_mean']:.4f}")
    print(f"Equity premium              : {res_no_habit['equity_premium']:.4f}")
    print()

    print("=== Strong habit (alpha = 0.8) ===")
    print(f"Average risk-free rate      : {res_habit['Rf_mean']:.4f}")
    print(f"Average risky asset return  : {res_habit['Re_mean']:.4f}")
    print(f"Equity premium              : {res_habit['equity_premium']:.4f}")
