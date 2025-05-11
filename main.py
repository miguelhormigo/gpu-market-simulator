import random

from config import *
from models.market import Market
from models.agent import RandomAgent, TrendFollowerAgent, AntiTrendAgent
from models.smart_agent import BuyFirstSellLastAgent


def simulate():
    market = Market()

    smart_agent = BuyFirstSellLastAgent(market)
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

    print(f"\n📊 Resultados Finales:")
    print(f"💰 Precio final del mercado: ${market.price:.2f}")
    print(f"📦 Stock restante en el mercado: {market.stock}")
    print(f"🧠 Balance del SmartAgent: ${smart_agent.balance:.2f}")
    print(f"🎮 GPUs en posesión del SmartAgent: {smart_agent.gpus} (debería ser 0)")

if __name__ == "__main__":
    simulate()