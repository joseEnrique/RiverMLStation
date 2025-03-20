from rivergenerator.movingwindow_river_generator import MovingWindowRiverGenerator
from rivergenerator.river_dataset_generator import RiverDatasetGenerator
from river import datasets, optim, metrics,compose, linear_model, preprocessing,feature_extraction, stats
from rivermultiproccesing.river_pipe import RiverModelManagerPipe
import time



def get_hour(x):
    x['hour'] = x['moment'].hour
    return x


if __name__ == "__main__":
    model = compose.Select('wind')
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
    print (dataset.take(1))
    start_time = time.time()
    generator = MovingWindowRiverGenerator(stream_period=0, dataset=dataset, n_instances=10,forecasting_horizon=2,past_history=2)
    for x, y in generator:
        print(len(x))

