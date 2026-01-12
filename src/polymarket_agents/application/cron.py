import time
from scheduler import Scheduler as JobScheduler
from scheduler.trigger import Monday

from polymarket_agents.application.trade import Trader  # adjust import after restructure


class TradingAgent:
    """Schedules and runs weekly trading jobs."""
    def __init__(self) -> None:
        self.trader = Trader()
        self.scheduler = JobScheduler()
        self.scheduler.weekly(Monday(), self.trader.one_best_trade)

    def start(self, poll_interval: float = 1.0) -> None:
        while True:
            self.scheduler.exec_jobs()
            time.sleep(poll_interval)