from datetime import datetime
import time

from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from keras.models import load_model
import numpy as np

from machine_learning.data_prep import DataPrep
from ExchangeRateData import ExchangeRateData
from display.Plots import GaiPlot

class RnnPredict:
    def train():
        price_items = ExchangeRateData.get_exchange_items("./db/BTC_EUR_echange_db.csv")
        price_items = ExchangeRateData.filter_exchange_items(price_items, datetime(2016, 1, 1), datetime(2022, 12, 30))
        prices = np.array([day.low+((day.high-day.low)/2) for day in price_items]).reshape(-1, 1)
        dates = np.array([day.date for day in price_items]).reshape(-1, 1)
        volumes = np.array([day.volume for day in price_items]).reshape(-1, 1)
        
        price_matrix, _, volume_matrix, price_matrix_scaled, volume_matrix_scaled = DataPrep.prepare_data_matrix(prices, dates, volumes) # Creating a matrix using the dataframe
        #combined_data = DataPrep.combine_data([price_matrix_scaled, volume_matrix_scaled])
        row, X_train, y_train, X_test, y_test = DataPrep.train_test_split_(price_matrix_scaled, train_size=.98) # Applying train-test splitting, also returning the splitting-point



        # LSTM Model parameters, I chose
        batch_size = 8            # Batch size (you may try different values)
        epochs = 70             # Epoch (you may try different values)
        seq_len = 30              # 30 sequence data (Representing the last 30 days)
        loss='mean_squared_error' # Since the metric is MSE/RMSE
        optimizer = 'adam'     # Recommended optimizer for RNN
        activation = 'linear'     # Linear activation
        input_shape=(None,1)      # Input dimension
        output_dim = 30           # Output dimension



        ##from book
        from keras import layers
        from keras.optimizers import RMSprop
        from tensorflow.keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM
        from tensorflow.python.keras.layers import CuDNNLSTM
        from tensorflow import keras
        SEQ_LEN = 30
        DROPOUT = 0.2
        WINDOW_SIZE = SEQ_LEN - 1
        BATCH_SIZE = 64

        DROPOUT = 0.1
        # model = Sequential()
        # # dropout=0.1, recurrent_dropout=0.5,
        # model.add(CuDNNLSTM(32, input_shape=input_shape, return_sequences=True))
        # model.add(CuDNNLSTM(64, return_sequences=True))
        # model.add(CuDNNLSTM(32))
        # model.add(layers.Dense(1))
        # model.compile(optimizer='adam', loss='mae', metrics=['mean_absolute_error'])

        ##from book
        from keras import layers
        from keras.optimizers import RMSprop
        model = Sequential()
        # dropout=0.1, recurrent_dropout=0.5,
        model.add(layers.LSTM(32, input_shape=input_shape, return_sequences=True))
        model.add(layers.LSTM(64, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss='mae', metrics=['mean_absolute_error'])


        start_time = time.time()
        history =model.fit(
            x=X_train,
            y=y_train,
            epochs=epochs,
            validation_split=0.05,
            batch_size=64
            )
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"proc time: {processing_time}")

        results = model.evaluate(X_test, y_test)


        import matplotlib.pyplot as plt
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()

        model.save('coin_predictor.h5')

    def test():
        price_items = ExchangeRateData.get_exchange_items("./db/BTC_EUR_echange_db.csv")
        price_items = ExchangeRateData.filter_exchange_items(price_items, datetime(2021, 3, 22), datetime(2021, 6, 28))
        prices = np.array([day.low+((day.high-day.low)/2) for day in price_items]).reshape(-1, 1)
        dates = np.array([day.date for day in price_items]).reshape(-1, 1)
        volumes = np.array([day.volume for day in price_items]).reshape(-1, 1)
        
        price_matrix, dates_matrix, volume_matrix, price_matrix_scaled, volume_matrix_scaled = DataPrep.prepare_data_matrix(prices, dates, volumes) # Creating a matrix using the dataframe
        #combined_data = DataPrep.combine_data([price_matrix_scaled, volume_matrix_scaled])
        #row, X_train, y_train, X_test, y_test = DataPrep.train_test_split_(price_matrix_scaled, train_size=.98) # Applying train-test splitting, also returning the splitting-point

        model = load_model('coin_predictor.h5')
        preds = model.predict(np.array(price_matrix_scaled,).reshape(len(price_matrix_scaled), -1, 1), batch_size=2)
        preds_rewind = []
        for i in range(len(preds)):
            pred_rewind = np.array(DataPrep.normalize_window_advanced_rewind(preds[i], price_matrix[i], 1.66166))
            preds_rewind.append(pred_rewind)
        GaiPlot.print_prediction_plot_compare(np.array(dates_matrix).reshape(-1), np.array(preds_rewind).reshape(-1))
        real_prices = []
        for window in price_matrix:
            real_prices.append(window[-1])
        #GaiPlot.print_forecast_deviation(np.array(dates_matrix).reshape(-1), np.array(real_prices) ,np.array(preds_rewind).reshape(-1))
        GaiPlot.print_forecast_deviation_relative_last_day(np.array(dates_matrix).reshape(-1) ,np.array(preds_rewind).reshape(-1))


if __name__ == "__main__":
    RnnPredict.test()