import datetime

import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataPrep:
    def combine_data(datalist):
        if len(datalist) < 2:
            return datalist
        return_list = []
        for i in range(len(datalist[0])):
            interval_list = []
            for j in range(len(datalist[0][0])):
                combined_values = []
                for k in range(len(datalist)):
                    combined_values.append(datalist[k][i][j])
                interval_list.append(combined_values)
            return_list.append(interval_list)
        return return_list

    def get_window_backwards(sequence_length):
        print("")


    def prepare_data_matrix(prices, dates, volumes, seq_len=30, scale_ranges=[True, True]):
        price_matrix = []
        price_matrix_scaled = []
        return_dates = []
        return_vols = []
        return_vols_scaled = []
        for index in range(len(prices)-seq_len+1):
            price_matrix.append(prices[index:index+seq_len])
            if scale_ranges[0] == True:
                prices_transformed = [((float(p[0]) / float(prices[index][0])) - 1) for p in prices[index:index+seq_len]]
                price_matrix_scaled.append(prices_transformed)
            return_dates.append(dates[index]+datetime.timedelta(days=seq_len))
            return_vols.append(volumes[index:index+seq_len])
            if scale_ranges[1] == True:
                scaler = MinMaxScaler(feature_range=(-1, 1))
                scaler = scaler.fit(volumes[index:index+seq_len])
                #volumes_transformed = scaler.transform(volumes[index:index+seq_len])
                volumes_transformed = [((float(p[0]) / float(volumes[index][0])) - 1) for p in volumes[index:index+seq_len]]
                return_vols_scaled.append(volumes_transformed)
        max_val_price = np.max(price_matrix_scaled) if np.max(price_matrix_scaled) > abs(np.min(price_matrix_scaled)) else abs(np.min(price_matrix_scaled))
        max_val_volume = np.max(return_vols_scaled) if np.max(return_vols_scaled) > abs(np.min(return_vols_scaled)) else abs(np.min(return_vols_scaled))
        print(f"max price {max_val_price} max vol {max_val_volume}") 
        price_matrix_scaled_final = np.array(price_matrix_scaled) / 1.66166 
        volume_matrix_scaled_final = np.array(return_vols_scaled) / 20    
        volume_matrix_scaled_final[volume_matrix_scaled_final>1.] = 1.
        volume_matrix_scaled_final[volume_matrix_scaled_final<-1.] = -1.    
        return price_matrix, return_dates, return_vols, price_matrix_scaled_final.tolist(), volume_matrix_scaled_final.tolist()        
        


    def price_matrix_creator(prices, dates, seq_len=30, ):
        '''
        It converts the series into a nested list where every item of the list contains historic prices of 30 days
        '''
        price_matrix = []
        return_dates = []
        for index in range(len(prices)-seq_len+1):
            price_matrix.append(np.array(prices[index:index+seq_len]).reshape(-1, 1))
            return_dates.append(np.array(dates[index]+datetime.timedelta(days=seq_len)).reshape(-1, 1))
        return price_matrix, return_dates

    def normalize_window_advanced(window):
        return [((float(p[0]) / float(window[0][0])) - 1) for p in window]
        # reference = window[0][0]
        # max_multiplied_limited = []
        # for value in window:
        #     if value[0] == reference:
        #         max_multiplied_limited.append(reference)
        #     elif value[0] < reference and reference / value[0] > max_multiplier:
        #         max_multiplied_limited.append(reference / max_multiplier)
        #     elif value[0] > reference and value[0] / reference > max_multiplier:
        #         max_multiplied_limited.append(reference * max_multiplier)
        #     else:
        #         max_multiplied_limited.append(value[0])
        # max_multiplied_limited = np.array(max_multiplied_limited).reshape(-1, 1)

        # return_list_scaled_relative_first = []
        # scaled_window = MinMaxScaler(feature_range=(-.5, .5)).fit(window).transform(window)
        # for data in scaled_window:
        #     return_list_scaled_relative_first.append(data[0]-scaled_window[0][0])
        # return return_list_scaled_relative_first

    def normalize_window_advanced_rewind(pred, window, max_multiplier):
        return (((window[0] * pred)) * max_multiplier)+window[0]

        
        




    def normalize_windows(window_data):
        '''
        It normalizes each value to reflect the percentage changes from starting point
        '''
        normalised_data = []
        for window in window_data:
            normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
            normalised_data.append(normalised_window)
        return normalised_data



    def normalize_rewind(preds, normalized_data):
        preds_original = []
        for index in range(0, len(preds)):
            pred = (preds[index])* normalized_data[index][0]
            preds_original.append(pred)
        return preds_original

    def train_test_split_(price_matrix, train_size=0.9, shuffle=False, return_row=True):
        '''
        It makes a custom train test split where the last part is kept as the training set.
        '''
        price_matrix = np.array(price_matrix)
        #print(price_matrix.shape)
        row = int(round(train_size * len(price_matrix)))
        train = price_matrix[:row, :]
        if shuffle==True:
            np.random.shuffle(train)
        X_train, y_train = train[:row,:-1], train[:row,-1]
        X_test, y_test = price_matrix[row:,:-1], price_matrix[row:,-1]
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        if return_row:
            return row, X_train, y_train, X_test, y_test
        else:
            X_train, y_train, X_test, y_test