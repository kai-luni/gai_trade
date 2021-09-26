import sys
#sys.path.append('C:\\Users\\kai_luni\\Dropbox\\kai\\code\\gai_t_b\\')

import matplotlib.pyplot as plt
import numpy as np

from ExchangeRateData import ExchangeRateData
from machine_learning.data_prep import DataPrep



class DataPrint:
    def print_diff_from_step_one(data_list : np.array):
        minimum_mp_list = []
        maximum_mp_list = []
        counter = 0
        for window in data_list:
            reference = window[0][0]
            minimum_value = np.amin(window)
            maximum_value = np.amax(window)
            if minimum_value == reference:
                minimum_mp_list.append(1.)
            else:
                min_factor = reference / minimum_value
                if min_factor > 20:
                    min_factor = 20
                minimum_mp_list.append(min_factor)

            if maximum_value == reference:              
                maximum_mp_list.append(1.)
            else:
                max_factor = maximum_value / reference
                if max_factor > 20:
                    max_factor = 20
                maximum_mp_list.append(max_factor)

        plt.hist(minimum_mp_list, density=False, bins=50)  # density=False would make counts
        plt.ylabel('Count')
        plt.xlabel('Minimum Multipyiers')
        plt.show()

        #clear figure
        plt.clf()

        plt.hist(maximum_mp_list, density=False, bins=50)  # density=False would make counts
        plt.ylabel('Count')
        plt.xlabel('Maximum Multiplyers')
        plt.show()




price_items = ExchangeRateData.get_exchange_items("./db/BTC_EUR_echange_db.csv")
prices = np.array([day.low+((day.high-day.low)/2) for day in price_items]).reshape(-1, 1)
dates = np.array([day.date for day in price_items]).reshape(-1, 1)
volumes = np.array([day.volume for day in price_items]).reshape(-1, 1)
price_matrix, _, volume_matrix, price_matrix_scaled, volume_matrix_scaled = DataPrep.prepare_data_matrix(prices, dates, volumes) # Creating a matrix using the dataframe

DataPrint.print_diff_from_step_one(volume_matrix)

# plt.hist([1,2,3,4], density=True, bins=50)  # density=False would make counts
# plt.xlim(0., 100.)
# plt.ylabel('Multiplier')
# plt.xlabel('Data')
# plt.show()