import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import six
from pyrqa.settings import Settings
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation
time_series = [0.1, 0.5, 0.3, 1.7, 0.8, 2.4, 0.6, 1.2, 1.4, 2.1, 0.8]
settings = Settings(time_series,
                        embedding_dimension=3,
                        time_delay=1,
                        neighbourhood=FixedRadius(1.0),
                        similarity_measure=EuclideanMetric,
                        theiler_corrector=1,
                        min_diagonal_line_length=2,
                        min_vertical_line_length=2,
                        min_white_vertical_line_length=2)
computation = RQAComputation.create(settings, verbose=True)
result = computation.run()
print result
# path = os.getcwd() + '\data.txt'  
# data = pd.read_csv(path, header=None, names=['Time', 'Passenger'])

# cols = data.shape[1]
# x = data.iloc[:,cols-1:cols]
# x = x.values
# rplt.RecurrencePlot(x, threshold=0.1)
