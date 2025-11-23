import torch
import torch.optim as optim
import random
import pandas as pd
import matplotlib.pyplot as plt  # <--- New Import
from trading_env import StockTradingEnv
from quantum_agent import QuantumDQN

# --- Settings ---
EPISODES = 20  # Graph acha dikhane ke liye thoda badha diya
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
model = QuantumDQN(input_shape=6, n_actions=3) 
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = torch.nn.MSELoss()

# --- Graph ke liye data store karne ki list ---
all_rewards = [] 

print("Training Start (Quantum simulation running)...")

for episode in range(EPISODES):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32)
    
    total_reward = 0
    done = False
    
    while not done:
        if random.random() < EPSILON:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = model(state.unsqueeze(0))
                action = torch.argmax(q_values).item()
        
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
        
    # --- Store Reward for Graph ---
    all_rewards.append(total_reward)
    
    EPSILON *= EPSILON_DECAY
    print(f"Episode {episode+1}/{EPISODES} completed. Total Profit: {total_reward:.2f}")

print("🎉 Training Finished!")

# --- Graph Plotting Logic ---
print("Generating Graph...")

plt.figure(figsize=(10, 5))
plt.plot(all_rewards, color='green', marker='o', linestyle='-')
plt.title('Quantum Bot Profit over Episodes')
plt.xlabel('Episode')
plt.ylabel('Total Profit')
plt.grid(True)

# Graph ko save karo taaki report me laga sako
plt.savefig('training_graph.png')
print("Graph saved as 'training_graph.png'")

# Graph ko screen par dikhao
plt.show()