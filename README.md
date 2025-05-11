# 🧠 GPU Market Simulator with Reinforcement Learning Agents

This project is a solution to a test. The test consists of developing a small object-oriented code project in Python. The context is based on an economy driven by the exchange of graphics cards, where a series of agents operate under certain rules.

The project models the following:
• A graphics card market with a limited stock of 100,000 units.
• A series of economic agents who buy and sell graphics cards in the market.

In each market iteration, all agents are ordered randomly, and one by one, they can choose to buy a card, sell a card, or do nothing. Each time an agent buys, the price of graphics cards increases by 0.5%. Each time an agent sells, the price decreases by 0.5%.

In total, there are 100 agents in the market, distributed as follows:
• 51 random agents who, in each iteration, have a 1/3 probability of buying, a 1/3 probability of selling, and a 1/3 probability of doing nothing.
• 24 trend-following agents who, in each iteration, have a 75% probability of buying and a 25% probability of doing nothing if the price has increased by 1% (or more) compared to the end of the previous iteration. Otherwise, they have a 20% probability of selling and an 80% probability of doing nothing.
• 24 anti-trend agents who, in each iteration, have a 75% probability of buying and a 25% probability of doing nothing if the price has decreased by 1% (or more) compared to the end of the previous iteration. Otherwise, they have a 20% probability of selling and an 80% probability of doing nothing.
• 1 agent with a logic defined by you, with the goal of maximizing their economic balance at the end of the simulation. This agent must end the last iteration with zero graphics cards in their possession.

Each agent starts with an initial balance of $1,000 and cannot borrow money. Agents cannot sell more cards than they possess. Each agent is aware that there are 100 other participating agents and how the buying and selling policies are distributed among the agent population. Each agent knows their turn within the market iteration but is unaware of the turns of the other agents.

The graphics card market starts with a unit price of $200.00, and from that point onwards, the agents begin to operate. A total of 1,000 iterations must be simulated.

## 📦 Project Structure

gpu-market-simulator/
├── models/
│   ├── agent.py              # Base Agent classes
│   ├── market.py             # Simulated GPU market
│   ├── q_agent.py            # Tabular Q-learning agent
│   ├── smart_agent.py        # Rule-based baseline agents
├── train_q_agent.py          # Script to train the Q-agent
├── main.py                   # Main class to simulate iterations
├── tests/
│   ├── test_simulation.py    # Simulation test
│   ├── test_agent_performance.py # Compare agents performance
├── README.md

---

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/miguelhormigo/gpu-market-simulator.git
cd gpu-market-simulator
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 🧪 Running Tests

```bash
PYTHONPATH=. pytest
```

## 🎯 Training the Q-learning Agent

```bash
python3 train_q_agent.py
```

The agent will:
	•	Interact with the market
	•	Learn from its actions via Q-learning
	•	Save its Q-table to q_table.pkl at the end

## 🤖 Agents

### ✅ TabularQAgent
	•	Uses a discretized state space (price change, inventory, time left)
	•	Learns via Q-learning
	•	Stores Q-values in a dictionary

### 🧻 BuyFirstSellLastAgent
	•	Buys GPUs early
	•	Sells all near the end
	•	Used as a simple baseline

## 📊 Evaluating Performance

You can compare agents using:

```bash
PYTHONPATH=. pytest tests/test_agent_performance.py
```

This test simulates both the Q agent and two baseline agents over many runs and compares average final balance.

## 💾 Saving & Loading Q-table

Q-tables are saved to q_table.pkl. You can load it later to continue training or for evaluation.