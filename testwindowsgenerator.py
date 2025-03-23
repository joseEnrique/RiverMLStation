from river.datasets import synth

from generator.movingwindow_river_generator import MovingWindowRiverGenerator
from generator.river_dataset_generator import RiverDatasetGenerator
from river import datasets, optim, metrics,compose, linear_model, preprocessing,feature_extraction, stats
from rivermultiproccesing.river_pipe import RiverModelManagerPipe
import time





if __name__ == "__main__":
    dataset = synth.FriedmanDrift(
        drift_type='lea',
        position=(2000, 5000, 8000),
        seed=123
    )
    # 2) Initialize the manager with a path for persistence
    #    If "my_model.pkl" exists, it will be loaded; else a new pipeline is used.
    generator = MovingWindowRiverGenerator(stream_period=0, dataset=dataset, n_instances=10,forecasting_horizon=3,past_history=2)
    for x, y in generator:
        print (y)
        pass
