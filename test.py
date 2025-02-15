import time
from river import compose, linear_model, preprocessing

from main import RiverModelManager

from river import datasets, optim, metrics,feature_extraction,stats


def get_hour(x):
    x['hour'] = x['moment'].hour
    return x


if __name__ == "__main__":
    # 1) Create any River pipeline or model
    #    We'll do a simple regression pipeline: StandardScaler + LinearRegression


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
    manager = RiverModelManager(model=model)
    dataset = datasets.Bikes()
    try:
        for x,y in dataset.take(1000):
            manager.train_one(x, y)

        y_preds = []
        for x, y in dataset.take(1000):
            y_pred = manager.predict_one(x)
            y_preds.append(y_pred)

        for y_pred, y in zip(y_preds,dataset.take(1000)):
            metric.update(y[1], y_pred)

        print (metric)


    finally:
        # 4) Stop the server, which also saves the model to "my_model.pkl"
        manager.stop()

        # Next time you run the script, it will detect "my_model.pkl", load it,
        # and continue from where it left off.
