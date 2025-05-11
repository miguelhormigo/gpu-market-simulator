from .agent import Agent


class BuyFirstSellLastAgent(Agent):
    def __init__(self, market):
        super().__init__(market, name="SmartAgent")
        self.history = []

    def act(self):
        # Estrategia: comprar todo lo que pueda al principio y 
        # vender lo más tarde posible, porque el precio siempre sube
        if self.market.get_pending_iterations_n() == self.gpus:
            self.sell()
            return

        if self.can_buy():
            self.buy()

class SmartAgent(Agent):
    def __init__(self, market):
        super().__init__(market, name="SmartAgent")
        self.history = []

    def act(self):
        # Si se va acercando la última ronda y aún tiene stock, liquidar
        if self.market.get_pending_iterations_n() == self.gpus:
            self.sell()
            return

        # Estrategia simple: comprar cuando ha bajado más de 1% y
        # vender si ha subido más de 1%
        change = self.market.get_price_change_pct()
        if change <= -0.01 and self.can_buy():
            self.buy()
        elif change >= 0.01 and self.can_sell():
            self.sell()
        else:
            pass  # no hace nada

    def liquidate(self):
        while self.can_sell():
            self.sell()