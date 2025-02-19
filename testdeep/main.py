from river import compose, preprocessing, metrics, evaluate
import time
from datetime import datetime as dt
from deep_river.regression import RollingRegressor

import torch
from river.datasets import synth
from tqdm import tqdm
from lstm import LSTMModule,NewLstmModule
import csv

_ = torch.manual_seed(42)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = synth.FriedmanDrift(
   drift_type='lea',
   position=(2000, 5000, 8000),
   seed=123
)

metric = metrics.MAE()


ae = RollingRegressor(module=NewLstmModule,
    loss_fn="mse",
    optimizer_fn="adam",
    window_size=500,
    lr=1e-2,
    device="cuda:0",
    hidden_size=64,  # parameters of MyModule can be overwritten
    append_predict=False,)


#ae = RollingRegressor(
#    module=LSTMTranslate,
#    loss_fn="mse",
#    optimizer_fn="adam",
#    window_size=500,
#    lr=1e-2,
#    device="cuda:0",
#    seq_length=500,
#    output_size=1,
#    return_sequences=True,
#    append_predict=False,
#)
#scaler = preprocessing.StandardScaler()



#print("timestamp,instances,metric")
start_time = time.time()
start_dt_object = dt.fromtimestamp(start_time)
start_formatted_time = start_dt_object.strftime('%Y-%m-%d %H:%M:%S.%f')
#print(f'{start_formatted_time},0,0')
count = 0
for x, y in (dataset.take(100000)):
    y_pred = ae.predict_one(x)
    metric.update(y_true=y, y_pred=y_pred)
    ae.learn_one(x=x, y =y)
    #print(f"MAE: {metric.get():.2f}")
    current_time = time.time()
    current_dt_object = dt.fromtimestamp(current_time)
    current_formatted_time = current_dt_object.strftime('%Y-%m-%d %H:%M:%S.%f')
    count += 1
    #if count % 10 == 0:
    #    print(f'{current_formatted_time},{count},{metric.get()}')

elapsed_time = time.time() - start_time
end_time = time.time()
end_dt_object = dt.fromtimestamp(end_time)
# Formatear el objeto datetime al formato deseado
end_formatted_time = end_dt_object.strftime('%Y-%m-%d %H:%M:%S.%f')

