#encoding=utf-8
import qlib
import pandas as pd
from qlib.constant import REG_US
from qlib.utils import exists_qlib_data, init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.utils import flatten_dict
import pickle
from pyecharts import options as opts
from pyecharts.charts import Kline

provider_uri = "../qlib_data/"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_US)

# train model
data_handler_config = {
    "start_time": "2021-09-01",
    "end_time": "2022-03-01",
    "fit_start_time": "2021-09-01",
    "fit_end_time": "2022-03-01",
    "instruments": "all",
    "learn_processors" : [],
    "infer_processors" : [],
    "freq" : "1min"
}

task = {
    "model": {
        "class": "XGBModel",
        "module_path": "qlib.contrib.model.xgboost",
        "kwargs": {
            "eval_metric": "rmse",
            "colsample_bytree": 0.8879,
            "eta": 0.0421,
            "max_depth": 20,
            "n_estimators": 647,
            "subsample": 0.8789,
            "nthread": 20
        },
    },
    "dataset": {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "MyAlphaHandler",
                "module_path": "my_data_handler",
                "kwargs": data_handler_config,
            },
            "segments": {
                # "train": ("2021-09-03", "2021-10-03"),
                # "valid": ("2021-10-04", "2021-11-03"),
                # "test": ("2021-11-04", "2021-12-01"),
                "train": ("2021-09-01", "2022-01-01"),
                "valid": ("2022-01-02", "2022-02-01"),
                "test": ("2022-02-02", "2022-03-01"),
            },
        },
    },
}

# model initiaiton
model = init_instance_by_config(task["model"])
dataset = init_instance_by_config(task["dataset"])

# start exp to train model
with R.start(experiment_name="train_model"):
    R.log_params(**flatten_dict(task))
    model.fit(dataset)
    R.save_objects(trained_model=model)
    rid = R.get_recorder().id
    print("rid: ", rid)

# # backtest and analysis
with R.start(experiment_name="backtest_analysis"):
    recorder = R.get_recorder(recorder_id=rid, experiment_name="train_model")
    model = recorder.load_object("trained_model")

    # prediction
    recorder = R.get_recorder()
    ba_rid = recorder.id
    sr = SignalRecord(model, dataset, recorder)
    sr.generate()

    port_analysis_config = {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "1min",
                "generate_portfolio_metrics": True,
            },
        },
        "strategy": {
            "class": "SimpleStrategy",
            "module_path": "simple_strategy",
            "kwargs": {
                "model": model,
                "dataset": dataset,
                "code" : "TSLA",
                "threshold" : 0.001,
                "hold_thresh" : 10,
            },
        },
        "backtest": {
            "start_time": "2022-01-04",
            "end_time": "2022-03-01",
            "account": 50000,
            "benchmark": 'TSLA',
            "exchange_kwargs": {
                "freq": "1min",
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0,
                "close_cost": 0,
                "min_cost": 2,
            },
        },
    }
    # backtest & analysis
    par = PortAnaRecord(recorder, port_analysis_config, "1min")
    par.generate()



# from qlib.contrib.report import analysis_model, analysis_position
# from qlib.data import D
# ba_rid = '8aa19f65244e45cfb2253e34b430cbdd'
# recorder = R.get_recorder(recorder_id=ba_rid, experiment_name="backtest_analysis")
# # # print(recorder)
# pred_df = recorder.load_object("pred.pkl")
# print(pred_df)
# pred_df_dates = pred_df.index.get_level_values(level='datetime')
# report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1min.pkl")
# positions = recorder.load_object("portfolio_analysis/positions_normal_1min.pkl")
# # print(positions)
# last = None
# buy, sell = 0, 0
# for key, value in positions.items():
#     pos = value.position
#     if 'TSLA' not in pos and last and 'TSLA' in last:
#         buy += 1
#         print("sell", last)
#     if 'TSLA' in pos and (not last or 'TSLA' not in last):
#         sell += 1
#         print('buy', pos)
#     last = pos
# print('buy:', buy, "sell:", sell)
    # print(key, value)
# analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1min.pkl")


# # # ## analysis position
# report_graph = analysis_position.report_graph(report_normal_df, show_notebook=False)
# for idx, fig in enumerate(report_graph):
#     fig.write_image("report%d.png" % idx)


# # ### risk analysis
# analysis_position.risk_analysis_graph(analysis_df, report_normal_df)


# # ## analysis model
# label_df = dataset.prepare("test", col_set="label")
# label_df.columns = ['label']


# # ### score IC
# pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)
# analysis_position.score_ic_graph(pred_label)

# # ### model performance
# analysis_model.model_performance_graph(pred_label)
