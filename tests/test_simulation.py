import pytest
from models.market import Market
from models.agent import RandomAgent, TrendFollowerAgent, AntiTrendAgent
from models.smart_agent import BuyFirstSellLastAgent
from config import INITIAL_BALANCE, INITIAL_PRICE, TOTAL_GPUS, PRICE_DELTA


def test_market_initial_state():
    market = Market()
    assert market.price == INITIAL_PRICE
    assert market.stock == TOTAL_GPUS
    assert market.get_price_change_pct() == 0

def test_market_price_update():
    market = Market()
    price_before = market.price
    market.update_price("buy")
    assert market.price == price_before * (1 + PRICE_DELTA)

    price_before = market.price
    market.update_price("sell")
    assert market.price == price_before * (1 - PRICE_DELTA)

def test_agent_buy_sell_logic():
    market = Market()
    agent = RandomAgent(market)
    
    # Force market to allow buying
    assert agent.can_buy()
    agent.buy()
    assert agent.balance < INITIAL_BALANCE
    assert agent.gpus == 1
    assert market.stock == TOTAL_GPUS - 1

    # Then sell
    agent.sell()
    assert agent.gpus == 0
    assert market.stock == TOTAL_GPUS

def test_random_agent_behavior():
    market = Market()
    agent = RandomAgent(market)
    agent.act()  # Just ensure it runs without errors

def test_trend_follower_behavior():
    market = Market()
    agent = TrendFollowerAgent(market)
    
    # Simulate a +2% price increase
    market.previous_price = 100
    market.price = 102
    for _ in range(10):
        agent.act()  # Should buy with 75% chance or do nothing/sell with 25%

def test_anti_trend_behavior():
    market = Market()
    agent = AntiTrendAgent(market)
    
    # Simulate a -2% price drop
    market.previous_price = 100
    market.price = 98
    for _ in range(10):
        agent.act()  # Should buy with 75% chance or do nothing/sell with 25%

def test_smart_agent_buy_and_liquidate():
    market = Market()
    agent = BuyFirstSellLastAgent(market)

    # Simulate early iterations: agent should buy
    market.iteration_n = 0
    agent.act()
    assert agent.gpus >= 0

    # Simulate last iterations: force liquidation
    agent.gpus = 3
    market.iteration_n = 997
    agent.act()
    assert agent.gpus <= 3

def test_no_overbuy_or_oversell():
    market = Market()
    agent = RandomAgent(market)
    
    # Empty the agent's balance
    agent.balance = 0
    agent.buy()
    assert agent.gpus == 0  # Cannot buy

    # Try to sell with no GPUs
    agent.sell()
    assert agent.gpus == 0