# models/q_agent.py
import random
from collections import defaultdict
from .agent import Agent


# Define actions as integers
ACTIONS = {0: "buy", 1: "sell", 2: "nothing"}

class TabularQAgent(Agent):
    def __init__(self, market, q_table=None):
        self.market = market
        self.balance = 1000
        self.gpus = 0
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1

        if q_table is None:
            self.q_table = defaultdict(self._default_q_values)
        else:
            self.q_table = q_table

        self.last_state = None
        self.last_action = None

    def _default_q_values(self):
        return {0: 0, 1: 0, 2: 0}

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(list(ACTIONS.keys()))  # return action index
        return max(self.q_table[state], key=self.q_table[state].get)

    def act(self):
        state = discretize_state(
            self.market.get_price_change_pct(),
            self.gpus,
            self.market.get_pending_iterations_n()
        )
        action = self.choose_action(state)

        # Interpret the action
        action_str = ACTIONS[action]
        if action_str == "buy" and self.can_buy():
            self.buy()
        elif action_str == "sell" and self.can_sell():
            self.sell()
        # "nothing" does nothing

        self.last_state = state
        self.last_action = action

    def learn(self, reward, next_state):
        q_old = self.q_table[self.last_state][self.last_action]
        max_next = max(self.q_table[next_state].values())
        new_q = q_old + self.alpha * (reward + self.gamma * max_next - q_old)
        self.q_table[self.last_state][self.last_action] = new_q

def discretize_state(price_change, gpus, it_left):
    pc = -1 if price_change <= -0.01 else 1 if price_change >= 0.01 else 0
    gpu = min(gpus, 10)
    it_bucket = 0 if it_left > 750 else 1 if it_left > 500 else 2 if it_left > 250 else 3
    return (pc, gpu, it_bucket)