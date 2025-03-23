from deep_river.regression import Regressor, RollingRegressor
from river import feature_extraction, compose, preprocessing,evaluate, metrics, datasets
from river import stats
import torch
import time
import copy
import datetime as dt

from generator.river_dataset_generator import RiverDatasetGenerator
from rivermultiproccesing.river_queue import RiverModelManager
from testdeep.lstm import NewLstmModule


def get_hour(x):
    x["hour"] = x["moment"].hour
    return x



if __name__ == "__main__":
    dataset = datasets.Bikes()
    model = compose.Select("clouds", "humidity", "pressure", "temperature", "wind")
    model += get_hour | feature_extraction.TargetAgg(
        by=["station", "hour"], how=stats.Mean()
    )
    model |= preprocessing.StandardScaler()
    model |= RollingRegressor(
        module=NewLstmModule,
        loss_fn="mse",
        optimizer_fn="adam",
        lr=1e-2,
        #device="cuda:0",
        hidden_size=64,
        window_size=3000,
    )
    metric = metrics.MAE()
    manager = RiverModelManager(model=copy.copy(model),training_device="cuda:0",predict_device="cuda:1")

    n_instances = 10000
#    start_time = time.time()
#    for x, y in RiverDatasetGenerator(stream_period=0, dataset=dataset, n_instances=n_instances):
#        y_pred = model.predict_one(x)
#        metric.update(y_true=y, y_pred=y_pred)
#        model.learn_one(x,y)
#    end_time = time.time()
#    elapsed_time = end_time - start_time
#    print(f"The Original Proccess took {elapsed_time} seconds.")
#    print(f"MAE: {metric.get():.2f}")

    start_time = time.time()
    for x, y in RiverDatasetGenerator(stream_period=0, dataset=dataset, n_instances=n_instances):
        print (x)
        y_pred = manager.predict_one(x)
        metric.update(y_true=y, y_pred=y_pred)
        manager.learn_one(x,y)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The Custom Proccess took {elapsed_time} seconds.")


    print(f"MAE: {metric.get():.2f}")
