# ⚛️ Quantum-Enhanced DQN Trading Bot

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![PennyLane](https://img.shields.io/badge/PennyLane-Quantum%20Computing-yellow)
![Status](https://img.shields.io/badge/Status-Active-success)

A **Hybrid Quantum-Classical Reinforcement Learning** agent designed for algorithmic trading. This project integrates **Variational Quantum Circuits (VQC)** with **Deep Q-Networks (DQN)** to optimize Buy/Sell strategies on financial data.

## 📌 Overview
Traditional classical neural networks often struggle with the complex, high-dimensional nature of financial market data. This project leverages **Quantum Machine Learning (QML)** to map input data into a high-dimensional Hilbert space, potentially improving the agent's ability to recognize market patterns.

The bot operates in a custom **Gymnasium** environment and learns to trade synthetic stock data to maximize profit over time.

## 🚀 Key Features
- **Hybrid Intelligence:** Combines Classical PyTorch Linear layers with PennyLane Quantum Entangler layers.
- **Quantum Encoding:** Uses `AngleEmbedding` to map market features into quantum states.
- **Reinforcement Learning:** Implements the DQN algorithm with Experience Replay (simplified).
- **Custom Environment:** A stock trading simulation built using OpenAI's `Gymnasium` interface.
- **Performance Visualization:** Automatically generates and saves profit graphs after training.

## 🛠️ Tech Stack
* **Language:** Python
* **Quantum Framework:** [PennyLane](https://pennylane.ai/)
* **Deep Learning:** [PyTorch](https://pytorch.org/)
* **RL Environment:** Gymnasium
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib

## 📂 Project Structure
```bash
QuantumTradingBot/
├── main.py             # Main training loop & Agent execution
├── quantum_agent.py    # Hybrid Quantum-Classical Neural Network Class
├── trading_env.py      # Custom Stock Trading Environment (Gym)
├── market_data.py      # Synthetic data generator
├── stock_data.csv      # Generated dataset (Input)
├── training_graph.png  # Output graph of agent's performance
└── README.md           # Project Documentation