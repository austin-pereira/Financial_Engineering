import numpy as np

def kalman_filter(prices):
    Q = 1e-5     # process variance
    R = 0.1      # measurement variance
    
    x = prices[0]   # initial estimate
    P = 1.0         # initial covariance
    
    estimates = []
    
    for z in prices:
        # Prediction
        P = P + Q
        
        # Kalman Gain
        K = P / (P + R)
        
        # Correction
        x = x + K * (z - x)
        P = (1 - K) * P
        
        estimates.append(x)
    
    return estimates

data = [100, 101, 130, 103, 104]
print(kalman_filter(data))
