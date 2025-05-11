import pickle
import random
from contextlib import suppress

from models.q_agent import TabularQAgent
from models.market import Market
from models.agent import RandomAgent, TrendFollowerAgent, AntiTrendAgent
from config import ITERATIONS


Q_TABLE_PATH = "q_table.pkl"

def load_q_table(path):
    with suppress(FileNotFoundError):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def save_q_table(q_table, path):
    with open(path, "wb") as f:
        pickle.dump(q_table, f)


def train_q_agent(runs=1000):
    q_table = load_q_table(Q_TABLE_PATH)

    for run in range(runs):
        market = Market()
        q_agent = TabularQAgent(market, q_table)
        agents = (
            [RandomAgent(market, name=f"Random_{i}") for i in range(50)] +
            [TrendFollowerAgent(market, name=f"Trend_{i}") for i in range(24)] +
            [AntiTrendAgent(market, name=f"Anti_{i}") for i in range(24)] +
            [q_agent]
        )

        for _ in range(ITERATIONS):
            market.reset_iteration()
            random.shuffle(agents)
            for agent in agents:
                agent.act()

        reward = q_agent.balance - (1000 * q_agent.gpus)
        q_agent.gpus = 0
        q_agent.learn(reward, (0, 0, 0))

        q_table = q_agent.q_table

        if run % 100 == 0:
            print(f"Run {run}: Balance = ${q_agent.balance:.2f}, GPUs = {q_agent.gpus}")
            save_q_table(q_table, Q_TABLE_PATH)

    save_q_table(q_table, Q_TABLE_PATH)


if __name__ == "__main__":
    train_q_agent(runs=1000)