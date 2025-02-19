from rivergenerator.dataset_generator import RiverDatasetGenerator
from river import datasets, optim, metrics,compose, linear_model, preprocessing,feature_extraction, stats
from rivermultiproccesing.river_pipe import RiverModelManagerPipe
import time



def get_hour(x):
    x['hour'] = x['moment'].hour
    return x


if __name__ == "__main__":
    model = compose.Select('clouds', 'humidity', 'pressure', 'temperature', 'wind')
    model += (
            get_hour |
            feature_extraction.TargetAgg(by=['station','hour'], how=stats.Mean())
    )
    model |= preprocessing.StandardScaler()
    model |= linear_model.LinearRegression(optimizer=optim.SGD(0.001))
    metric = metrics.MAE()
    # 2) Initialize the manager with a path for persistence
    #    If "my_model.pkl" exists, it will be loaded; else a new pipeline is used.

    dataset = datasets.Bikes()

    start_time = time.time()
    generator = RiverDatasetGenerator(stream_period=0, dataset=dataset, n_instances=dataset.n_outputs)
    for x, y in generator:
        y_pred = model.predict_one(x)
        model.learn_one(x, y)
        metric.update(y_pred, y)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(metric)
    print(f"The Original Proccess took {elapsed_time} seconds.")
    print (generator.get_count())

    anothermodel = compose.Select('clouds', 'humidity', 'pressure', 'temperature', 'wind')
    anothermodel += (
            get_hour |
            feature_extraction.TargetAgg(by=['station','hour'], how=stats.Mean())
    )
    anothermodel |= preprocessing.StandardScaler()
    anothermodel |= linear_model.LinearRegression(optimizer=optim.SGD(0.001))
    metric = metrics.MAE()
    manager = RiverModelManagerPipe(model=anothermodel)
    start_time = time.time()
    for x,y in RiverDatasetGenerator(stream_period=0,dataset=dataset,n_instances=100000000000000000):
        y_pred = manager.predict_one(x)
        manager.learn_one(x, y)
        metric.update(y_pred, y)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(metric)
    print(f"The Custom Process took {elapsed_time} seconds.")

    manager.stop()

