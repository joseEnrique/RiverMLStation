from RiverGenerator.dataset_generator import RiverDatasetGenerator
from river import datasets, optim, metrics,compose, linear_model, preprocessing
from main import RiverModelManager
import time




if __name__ == "__main__":
    model = compose.Select('clouds', 'humidity', 'pressure', 'temperature', 'wind')

    model |= preprocessing.StandardScaler()
    model |= linear_model.LinearRegression(optimizer=optim.SGD(0.001))
    metric = metrics.MAE()
    # 2) Initialize the manager with a path for persistence
    #    If "my_model.pkl" exists, it will be loaded; else a new pipeline is used.

    dataset = datasets.Bikes()

    start_time = time.time()
    for x, y in RiverDatasetGenerator(stream_period=0, dataset=dataset, n_instances=1000000):
        y_pred = model.predict_one(x)
        model.learn_one(x, y)
        metric.update(y_pred, y)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(metric)
    print(f"The Original Proccess took {elapsed_time} seconds.")

    anothermodel = compose.Select('clouds', 'humidity', 'pressure', 'temperature', 'wind')
    anothermodel |= preprocessing.StandardScaler()
    anothermodel |= linear_model.LinearRegression(optimizer=optim.SGD(0.001))
    metric = metrics.MAE()
    manager = RiverModelManager(model=anothermodel)
    start_time = time.time()
    for x,y in RiverDatasetGenerator(stream_period=0,dataset=dataset,n_instances=1000000):
        y_pred = manager.predict_one(x)
        manager.learn_one(x, y)
        metric.update(y_pred, y)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The Custom Process took {elapsed_time} seconds.")

    manager.stop()

