import pandas as pd
import numpy as np

def get_data():
    print("Generating synthetic market data...")
    
    days = 365
    dates = pd.date_range(start='2023-01-01', periods=days)
    
    prices = [100]
    for _ in range(days-1):
        change = np.random.uniform(-2, 2) 
        prices.append(prices[-1] + change)
        
    df = pd.DataFrame(data={'Close': prices}, index=dates)
    
    
    df['Normalized'] = (df['Close'] - df['Close'].min()) / (df['Close'].max() - df['Close'].min())
    
    return df

if __name__ == "__main__":
    df = get_data()
    df.to_csv("stock_data.csv")
    print("SUCCESS: Data saved to stock_data.csv")