import numpy as np
from sklearn.preprocessing import MinMaxScaler


test_list_one = np.array([1., 2., 3., 4.]).reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(-.5, .5))
scaler = scaler.fit(test_list_one)
result = scaler.transform(test_list_one)
final_list = []
for val in result:
    final_list.append(val - result[0])
print(final_list)

