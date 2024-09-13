import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
def generate_dummy_stock_data(days=30, ticks_per_day=78):
    data = []
    for day in range(days):
        start_time = pd.Timestamp(f'2024-01-{day+1:02d} 09:30:00')
        time_index = pd.date_range(start=start_time, periods=ticks_per_day, freq='5T')
        day_data = pd.DataFrame({
            'timestamp': time_index,
            'open': np.random.rand(len(time_index)) * 100,
            'high': np.random.rand(len(time_index)) * 100,
            'low': np.random.rand(len(time_index)) * 100,
            'close': np.random.rand(len(time_index)) * 100,
            'volume': np.random.randint(1000, 10000, len(time_index))
        })
        data.append(day_data)
    return pd.concat(data).reset_index(drop=True)

historical_data = generate_dummy_stock_data()

initial_balance = 10000.00 
balance = initial_balance
stock_held = 0 
transaction_fee = 10.00 

scaler = StandardScaler()

def backtest(model1, model2, historical_data, balance, stock_held):
    for day in range(30):
        day_data = historical_data.iloc[day*78:(day+1)*78].copy()
        
        for i in range(1, len(day_data)):
            X_model1 = day_data.iloc[:i, :-1].values 
            X_model1_scaled = scaler.transform(X_model1)

            predicted_prices = model1.predict(X_model1_scaled)

            current_input = np.hstack((
                day_data.iloc[i-1:i, :-1].values.flatten(),  
                predicted_prices[0], 
                np.array([0]),  
                np.array([balance - initial_balance])
            )).reshape(1, -1)
            current_input_scaled = scaler.transform(current_input)
            
            action = np.argmax(model2.predict(current_input_scaled))
            
            current_price = day_data.iloc[i]['close']
            
            if action == 1:  
                if balance > current_price + transaction_fee:
                    stock_held += 1
                    balance -= current_price + transaction_fee
                    print(f"Day {day+1}, Tick {i}: Bought 1 stock at {current_price:.2f}. Balance: {balance:.2f}, Stocks held: {stock_held}")
            
            elif action == 2:  
                if stock_held > 0:
                    stock_held -= 1
                    balance += current_price - transaction_fee
                    print(f"Day {day+1}, Tick {i}: Sold 1 stock at {current_price:.2f}. Balance: {balance:.2f}, Stocks held: {stock_held}")
            
    final_value = balance + stock_held * historical_data.iloc[-1]['close']
    profit_loss = final_value - initial_balance
    print(f"Final balance after 30 days: {final_value:.2f}")
    print(f"Total profit/loss: {profit_loss:.2f}")

