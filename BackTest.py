import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load or create your models
# model1 = ...  # Load the first model (stock price prediction model)
# model2 = ...  # Load the second model (decision-making model)

# Load historical stock data for the last 30 days (replace with actual data loading)
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

# Backtest parameters
initial_balance = 10000.00  # Starting with $10,000
balance = initial_balance
stock_held = 0  # Number of stocks held
transaction_fee = 10.00  # Transaction fee per trade

# Prepare your data
scaler = StandardScaler()

# Function to simulate trading over the past 30 days using both models
def backtest(model1, model2, historical_data, balance, stock_held):
    for day in range(30):
        # Get the data for the current day
        day_data = historical_data.iloc[day*78:(day+1)*78].copy()
        
        # Iterate over each tick in the day
        for i in range(1, len(day_data)):
            # Prepare the input data for model1
            X_model1 = day_data.iloc[:i, :-1].values  # Take all ticks up to this point
            X_model1_scaled = scaler.transform(X_model1)

            # Predict the rest of the day's prices using model1
            predicted_prices = model1.predict(X_model1_scaled)

            # Prepare the input data for model2 (decision-making)
            current_input = np.hstack((
                day_data.iloc[i-1:i, :-1].values.flatten(),  # Original features
                predicted_prices[0],  # Predicted price
                np.array([0]),  # Placeholder for action taken (set to 0)
                np.array([balance - initial_balance])  # Profit/loss
            )).reshape(1, -1)
            current_input_scaled = scaler.transform(current_input)
            
            # Use model2 to predict the best action (0 = hold, 1 = buy, 2 = sell)
            action = np.argmax(model2.predict(current_input_scaled))
            
            current_price = day_data.iloc[i]['close']
            
            if action == 1:  # Buy
                if balance > current_price + transaction_fee:
                    stock_held += 1
                    balance -= current_price + transaction_fee
                    print(f"Day {day+1}, Tick {i}: Bought 1 stock at {current_price:.2f}. Balance: {balance:.2f}, Stocks held: {stock_held}")
            
            elif action == 2:  # Sell
                if stock_held > 0:
                    stock_held -= 1
                    balance += current_price - transaction_fee
                    print(f"Day {day+1}, Tick {i}: Sold 1 stock at {current_price:.2f}. Balance: {balance:.2f}, Stocks held: {stock_held}")
            
            # Otherwise, hold

    # Calculate final value
    final_value = balance + stock_held * historical_data.iloc[-1]['close']
    profit_loss = final_value - initial_balance
    print(f"Final balance after 30 days: {final_value:.2f}")
    print(f"Total profit/loss: {profit_loss:.2f}")

# Assuming you have the trained models model1 and model2
# backtest(model1, model2, historical_data, balance, stock_held)
