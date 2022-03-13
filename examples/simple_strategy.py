# 需要写一个简单的交易策略
from qlib.contrib.strategy.signal_strategy import BaseSignalStrategy
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
import pandas as pd
import copy


class SimpleStrategy(BaseSignalStrategy):
    def __init__(
        self,
        *,
        code,
        threshold,
        hold_thresh,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.code = code
        self.threshold = threshold
        self.hold_thresh = hold_thresh
        
    def generate_trade_decision(self, execute_result=None):
        # get the number of trading step finished, trade_step can be [0, 1, 2, ..., trade_len - 1]
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        # print(self.signal.signal_cache)
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        # print(pred_start_time, pred_end_time, pred_score)
        # NOTE: the current version of topk dropout strategy can't handle pd.DataFrame(multiple signal)
        # So it only leverage the first col of signal
#         if isinstance(pred_score, pd.DataFrame):
#             pred_score = pred_score.iloc[:, 0]
        if pred_score is None:
            return TradeDecisionWO([], self)

        current_temp = copy.deepcopy(self.trade_position)
        # generate order list for this adjust date
        sell_order_list = []
        buy_order_list = []

        # load score
        cash = current_temp.get_cash()
        current_stock_list = current_temp.get_stock_list()
        stock_score = pred_score[self.code]

        time_per_step = self.trade_calendar.get_freq()

        call_sell = False
        if self.code in current_stock_list:
            price = current_temp.get_stock_price(self.code)
            sell_price = self.trade_exchange.get_deal_price(
                stock_id=self.code, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.SELL
            )
            # print("check:", price, sell_price)
            if sell_price >= price * 1.01 or sell_price < price * 0.995 or stock_score < -self.threshold:
                call_sell = True
            if current_temp.get_stock_count(self.code, bar=time_per_step) >= self.hold_thresh:
                call_sell = True
        print(current_temp, stock_score)
        if call_sell:
            # sell order
            sell_amount = current_temp.get_stock_amount(code=self.code)
            factor = self.trade_exchange.get_factor(
                stock_id=self.code, start_time=trade_start_time, end_time=trade_end_time
            )
            # sell_amount = self.trade_exchange.round_amount_by_trade_unit(sell_amount, factor)
            sell_order = Order(
                stock_id=self.code,
                amount=sell_amount,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=Order.SELL,  # 0 for sell, 1 for buy
            )
            # is order executable
            if self.trade_exchange.check_order(sell_order):
                sell_order_list.append(sell_order)
                trade_val, trade_cost, trade_price = self.trade_exchange.deal_order(
                    sell_order, position=current_temp
                )
        if stock_score > self.threshold:
            value = cash * self.risk_degree
            # buy order
            buy_price = self.trade_exchange.get_deal_price(
                stock_id=self.code, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.BUY
            )
            buy_amount = value / buy_price
            factor = self.trade_exchange.get_factor(stock_id=self.code, start_time=trade_start_time, end_time=trade_end_time)
            buy_amount = self.trade_exchange.round_amount_by_trade_unit(buy_amount, factor)
            buy_order = Order(
                stock_id=self.code,
                amount=buy_amount,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=Order.BUY,  # 1 for buy
            )
            buy_order_list.append(buy_order)
        return TradeDecisionWO(sell_order_list + buy_order_list, self)