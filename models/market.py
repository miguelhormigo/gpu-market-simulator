from config import *


class Market:
    def __init__(self):
        self.price = INITIAL_PRICE
        self.stock = TOTAL_GPUS
        self.previous_price = INITIAL_PRICE
        self.iteration_n = 0

    def update_price(self, action):
        if action == "buy":
            self.price *= (1 + PRICE_DELTA)
        elif action == "sell":
            self.price *= (1 - PRICE_DELTA)

    def reset_iteration(self):
        self.previous_price = self.price
        self.iteration_n += 1

    def get_price_change_pct(self):
        return (self.price - self.previous_price) / self.previous_price

    def get_pending_iterations_n(self):
        return ITERATIONS - self.iteration_n