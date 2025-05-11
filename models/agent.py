import random

from config import *


class Agent:
    def __init__(self, market, name="Agent"):
        self.market = market
        self.balance = INITIAL_BALANCE
        self.gpus = 0
        self.name = name

    def can_buy(self):
        return self.balance >= self.market.price and self.market.stock > 0

    def can_sell(self):
        return self.gpus > 0

    def buy(self):
        if self.can_buy():
            self.balance -= self.market.price
            self.gpus += 1
            self.market.stock -= 1
            self.market.update_price("buy")

    def sell(self):
        if self.can_sell():
            self.balance += self.market.price
            self.gpus -= 1
            self.market.stock += 1
            self.market.update_price("sell")

    def act(self):
        raise NotImplementedError

class RandomAgent(Agent):
    def act(self):
        choice = random.choice(["buy", "sell", "nothing"])
        if choice == "buy":
            self.buy()
        elif choice == "sell":
            self.sell()

class TrendFollowerAgent(Agent):
    def act(self):
        change = self.market.get_price_change_pct()
        if change >= 0.01 and random.random() < 0.75:
                self.buy()
        elif random.random() < 0.20:
                self.sell()

class AntiTrendAgent(Agent):
    def act(self):
        change = self.market.get_price_change_pct()
        if change <= -0.01 and random.random() < 0.75:
                self.buy()
        elif random.random() < 0.20:
                self.sell()