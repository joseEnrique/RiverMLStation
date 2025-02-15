from RiverGenerator.dataset_generator import RiverDatasetGenerator
from RiverGenerator.base_generator import BaseGenerator
from river import datasets, optim, metrics,compose, linear_model, preprocessing



model = compose.Select('clouds', 'humidity', 'pressure', 'temperature', 'wind')

model |= preprocessing.StandardScaler()
model |= linear_model.LinearRegression(optimizer=optim.SGD(0.001))
metric = metrics.MAE()
# 2) Initialize the manager with a path for persistence
#    If "my_model.pkl" exists, it will be loaded; else a new pipeline is used.

dataset = datasets.Bikes()


for i in RiverDatasetGenerator(stream_period=600,dataset=dataset,n_instances=100):
    print (i)


