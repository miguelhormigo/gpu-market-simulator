import random
import statistics

from config import *
from models.market import Market
from models.agent import RandomAgent, TrendFollowerAgent, AntiTrendAgent
from models.smart_agent import SmartAgent, BuyFirstSellLastAgent


NUM_RUNS = 100

def run_simulation(smart_agent_cls):
    market = Market()
    smart_agent = smart_agent_cls(market)
    agents = (
        [RandomAgent(market, name=f"RandomAgent_{i}") for i in range(51)] +
        [TrendFollowerAgent(market, name=f"Trend_{i}") for i in range(24)] +
        [AntiTrendAgent(market, name=f"AntiTrend_{i}") for i in range(24)] +
        [smart_agent]
    )

    for _ in range(ITERATIONS):
        market.reset_iteration()
        random.shuffle(agents)
        for agent in agents:
            agent.act()

    return smart_agent.balance

def test_agent_performance_comparison():
    print("\n⏱️ Comparing agent performance...")

    balances_a = [run_simulation(SmartAgent) for _ in range(NUM_RUNS)]
    balances_b = [run_simulation(BuyFirstSellLastAgent) for _ in range(NUM_RUNS)]

    def summarize(label, data):
        print(f"\n📈 {label}")
        print(f"Runs: {NUM_RUNS}")
        print(f"Average Balance: ${statistics.mean(data):.2f}")
        print(f"Median Balance: ${statistics.median(data):.2f}")
        print(f"Best: ${max(data):.2f}")
        print(f"Worst: ${min(data):.2f}")
        print(f"Std Dev: ${statistics.stdev(data):.2f}")

    summarize("SmartAgent", balances_a)
    summarize("BuyFirstSellLastAgent", balances_b)

    assert all(b >= 0 for b in balances_a), "SmartAgent negative balances"
    assert all(b >= 0 for b in balances_b), "BuyFirstSellLastAgent negative balances"