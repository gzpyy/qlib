#encoding=utf-8
import qlib
import pandas as pd
import pickle
import xgboost as xgb
import numpy as np
import re
from qlib.constant import REG_US
from qlib.utils import exists_qlib_data, init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.utils import flatten_dict
from qlib.data import LocalExpressionProvider
from qlib.data.ops import Operators, OpsList
from qlib.data.base import Feature
from pyecharts import options as opts
from pyecharts.charts import Kline, Line, Grid
from my_data_handler import MyAlphaHandler

# model_file = r'.\mlruns\1\d6536b056ba84a74be6b33971f443cf6\artifacts\trained_model'
model_file = r'.\mlruns\1\148ef1cd7acd48deac3eadc339ad3008\artifacts\trained_model'
with open(model_file, 'rb') as fi:
    model = pickle.load(fi)

exprs, columns = MyAlphaHandler.get_custom_config()


raw_data = pd.read_csv('../stock_data/TSLA.csv', parse_dates=['time'])
raw_data['data_time'] = raw_data['time'].dt.strftime("%Y-%m-%d %H:%M:00")
raw_data.set_index('time', inplace=True)
raw_data["vwap"] = np.nan
raw_data.sort_index(inplace=True)
# print(raw_data)
class MyFeature(Feature):
    def _load_internal(self, instrument, start_index, end_index, freq):
        print("load", self._name, instrument, start_index, end_index, freq)
        return raw_data.loc[start_index:end_index][self._name]
Operators.register(OpsList + [MyFeature])

def my_parse_field(field):
    if not isinstance(field, str):
        field = str(field)
    for pattern, new in [(r"\$(\w+)", rf'MyFeature("\1")'), (r"(\w+\s*)\(", r"Operators.\1(")]:  # Features  # Operators
        field = re.sub(pattern, new, field)
    return field

obj = dict()
for field in exprs:
    expression = eval(my_parse_field(field))
    series = expression.load('TSLA', "2022-01-02", "2022-02-28", "1min")
    series = series.astype(np.float32)
    obj[field] = series
data = pd.DataFrame(obj)
data.columns = columns

view_time_start = '2022-02-11'
view_time_end = '2022-02-12'

pre_data = raw_data.loc[view_time_start:view_time_end].copy()
pred=model.model.predict(xgb.DMatrix(data.loc[view_time_start:view_time_end]))
pre_data['pred_score'] = pred
records = pre_data.to_dict("records")

cash = 50000
position = {}
hold_thresh = 5
score_thresh = 0.001
x_axises, y_axises, mark_points, money = [], [], [], []
for record in records:
    x_axises.append(record['data_time'])
    y_axises.append([
        record['open'], record['close'], record['low'], record['high']
    ])
    if 'hold_cnt' in position:
        position['hold_cnt'] += 1
    if position and (record['open'] >= position['close'] * 1.01 or record['open'] < position['close'] * 0.995 or record['pred_score'] < -score_thresh or position['hold_cnt'] >= hold_thresh):
        cash += position['amount'] * record['open']
        position = {}
        #print("sell")
        mark_points.append(opts.MarkPointItem(
            coord=[record['data_time'], record['high']],
            symbol='triangle', symbol_size=7, 
            itemstyle_opts=opts.ItemStyleOpts(color="green")
        ))
    elif record['pred_score'] > score_thresh and not position:
        position = dict(record)
        position['amount'] = int(cash / position['open'])
        cash -= position['amount'] * position['open']
        # buy
        #print("buy")
        position['hold_cnt'] = 0
        mark_points.append(opts.MarkPointItem(
            coord=[record['data_time'], record['high']],
            symbol='arrow', symbol_size=7,
            itemstyle_opts=opts.ItemStyleOpts(color="yellow")
        ))
    cur_money = cash
    if position:
        cur_money += position['amount'] * record['close']
    money.append(cur_money)
if position:
    cash += position['amount'] * records[-1]['close']
print("cash:", cash)

kline_graph = (
    Kline()
    .add_xaxis(x_axises)
    .add_yaxis(
        "kline",
        y_axises,        
        markpoint_opts=opts.MarkPointOpts(
            data=mark_points
        ),
    )
    .set_global_opts(
        xaxis_opts=opts.AxisOpts(is_scale=True),
        yaxis_opts=opts.AxisOpts(
            is_scale=True,
            splitarea_opts=opts.SplitAreaOpts(
                is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
            ),
        ),
        title_opts=opts.TitleOpts(title="%s_%s" % (view_time_start, view_time_end)),
        datazoom_opts=[opts.DataZoomOpts(type_="inside", xaxis_index=[0, 1],)],
    )
)

kline_line = (
    Line()
    .add_xaxis(xaxis_data=x_axises)
    .add_yaxis(
        series_name="cur_money",
        y_axis=money,
        is_smooth=True,
        linestyle_opts=opts.LineStyleOpts(opacity=0.5),
        label_opts=opts.LabelOpts(is_show=False),
        markline_opts=opts.MarkLineOpts(
            data=[opts.MarkLineItem(y=50000)]
        ),
    )
    .set_global_opts(
        xaxis_opts=opts.AxisOpts(
            type_="category",
            grid_index=2,
            axislabel_opts=opts.LabelOpts(is_show=False),            
        ),
        yaxis_opts=opts.AxisOpts(
            min_='dataMin'
        )
    )
)

grid_chart = Grid(init_opts=opts.InitOpts(width='2000px', height='900px'))
grid_chart.add(
    kline_graph,
    grid_opts=opts.GridOpts(pos_left="3%", pos_right="10%", height="50%"),
)
grid_chart.add(
    kline_line,
    grid_opts=opts.GridOpts(
        pos_left="3%", pos_right="10%", pos_top="60%", height="30%"
    ),
)
grid_chart.render("kline_markline.html")