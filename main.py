import torch
import torch.optim as optim
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from trading_env import StockTradingEnv
from quantum_agent import QuantumDQN

# --- Settings ---
EPISODES = 20   # Training ke liye episodes
LR = 0.01
GAMMA = 0.9
EPSILON = 1.0
EPSILON_DECAY = 0.9

# --- Load Data ---
try:
    df = pd.read_csv("stock_data.csv")
    print("Data Loaded Successfully!")
except FileNotFoundError:
    print("Error: stock_data.csv nahi mili. Pehle market_data.py run karo.")
    exit()

env = StockTradingEnv(df)
# Input shape 6 hai (5 din ka price + 1 holding status)
model = QuantumDQN(input_shape=6, n_actions=3) 
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = torch.nn.MSELoss()

# --- Graph ke liye profit store karne ki list ---
all_rewards = [] 

print("Training Start (Quantum simulation running)...")

# ===========================
# === MAIN TRAINING LOOP ===
# ===========================
for episode in range(EPISODES):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32)
    
    total_reward = 0
    done = False
    
    while not done:
        if random.random() < EPSILON:
            action = env.action_space.sample() # Explore
        else:
            with torch.no_grad():
                q_values = model(state.unsqueeze(0))
                action = torch.argmax(q_values).item() # Exploit
        
        next_state, reward, done, _, _ = env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        
        # Target calculation
        with torch.no_grad():
            target_val = model(next_state.unsqueeze(0))
            max_next_q = torch.max(target_val).item()
            
        target = reward + GAMMA * max_next_q * (1 - int(done))
        
        current_q = model(state.unsqueeze(0))[0][action]
        
        target_tensor = torch.tensor(target, dtype=torch.float32)
        
        loss = loss_fn(current_q, target_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        state = next_state
        total_reward += reward
        
    # --- Store Reward for Profit Graph ---
    all_rewards.append(total_reward)
    
    EPSILON *= EPSILON_DECAY
    print(f"Episode {episode+1}/{EPISODES} completed. Total Profit: {total_reward:.2f}")

print("🎉 Training Finished!")

# ==========================================
# === NEW: Market Prediction Analysis Graph ===
# ==========================================
print("Generating Prediction Analysis Graph...")

# Hum data ka ek tukda lenge (e.g., Day 50 se Day 150 tak) analyze karne ke liye
start_day = 50
end_day = 150
analysis_df = df.iloc[start_day:end_day].copy().reset_index(drop=True)

# Lists to store data for plotting
plot_prices = []
q_hold = []
q_buy = []
q_sell = []

# Environment ko temporary reset karte hain analysis ke liye
env.df = analysis_df
state, _ = env.reset()
state = torch.tensor(state, dtype=torch.float32)
done = False

while not done:
    # Current actual price store karo (Normalized nahi, asli wala for visualization)
    current_step_idx = env.current_step
    # Thoda complex tareeka current price nikalne ka kyunki env normalized use karta hai
    actual_price = analysis_df['Close'].iloc[current_step_idx]
    plot_prices.append(actual_price)
    
    # Model se pucho: "Is state me teeno actions ki value kya hai?"
    with torch.no_grad():
        q_values = model(state.unsqueeze(0))
        q_values_np = q_values.detach().numpy().flatten()
        
        # Q-Values store karo
        q_hold.append(q_values_np[0])
        q_buy.append(q_values_np[1])
        q_sell.append(q_values_np[2])

    # Best action lo (sirf simulation ke liye)
    action = torch.argmax(q_values).item()
    next_state, _, done, _, _ = env.step(action)
    state = torch.tensor(next_state, dtype=torch.float32)

# --- Plotting the Prediction Graph ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Subplot 1: Asli Market Price
days = range(start_day, start_day + len(plot_prices))
ax1.plot(days, plot_prices, label='Actual Market Price', color='black', linewidth=2)
ax1.set_ylabel('Stock Price')
ax1.set_title('Market Price vs. Quantum Bot Predictions (Q-Values)')
ax1.legend()
ax1.grid(True)

# Subplot 2: Bot ki Soch (Q-Values)
ax2.plot(days, q_hold, label='Hold Q-Value', color='blue', linestyle='--')
ax2.plot(days, q_buy, label='Buy Q-Value', color='green', linewidth=2)
ax2.plot(days, q_sell, label='Sell Q-Value', color='red', linestyle='--')
ax2.set_ylabel('Q-Value (Predicted Value of Action)')
ax2.set_xlabel('Day')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('prediction_analysis.png')
print("Prediction Graph saved as 'prediction_analysis.png'")
plt.show()

# ===================================
# === Original Profit Graph (Optional now) ===
# ===================================
# Agar purana profit graph bhi chahiye toh ye uncomment
# plt.figure(figsize=(10, 5))
# plt.plot(all_rewards, color='purple', marker='o')
# plt.title('Total Profit per Episode during Training')
# plt.xlabel('Episode')
# plt.ylabel('Total Profit')
# plt.grid(True)
# plt.savefig('training_profit.png')
# plt.show()