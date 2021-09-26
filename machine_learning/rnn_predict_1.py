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
        prices = [day.low+((day.high-day.low)/2) for day in price_items]
        dates = [day.date for day in price_items]
        price_matrix, _ = DataPrep.price_matrix_creator(prices, dates) # Creating a matrix using the dataframe
        price_matrix = DataPrep.normalize_windows(price_matrix) # Normalizing its values to fit to RNN
        row, X_train, y_train, X_test, y_test = DataPrep.train_test_split_(price_matrix, train_size=.98) # Applying train-test splitting, also returning the splitting-point

        # LSTM Model parameters, I chose
        batch_size = 8            # Batch size (you may try different values)
        epochs = 100               # Epoch (you may try different values)
        seq_len = 30              # 30 sequence data (Representing the last 30 days)
        loss='mean_squared_error' # Since the metric is MSE/RMSE
        optimizer = 'adam'     # Recommended optimizer for RNN
        activation = 'linear'     # Linear activation
        input_shape=(None,1)      # Input dimension
        output_dim = 30           # Output dimension

        # model = Sequential()
        # model.add(LSTM(units=output_dim, return_sequences=True, input_shape=input_shape))
        # model.add(Dense(units=32,activation=activation))
        # model.add(LSTM(units=output_dim, return_sequences=False))
        # model.add(Dense(units=1,activation=activation))
        # model.compile(optimizer=optimizer,loss=loss, metrics=['mean_absolute_error'])

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
        #print(f"{model.metrics_names}:{results}")
        #print(f"history {history.history}")


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
        price_items = ExchangeRateData.filter_exchange_items(price_items, datetime(2020, 5, 1), datetime(2020, 6, 29))
        dates = [day.date for day in price_items]
        prices = [day.low+((day.high-day.low)/2) for day in price_items]

        price_matrix, dates_matrix = DataPrep.price_matrix_creator(prices, dates)
        X_test = DataPrep.normalize_windows(price_matrix)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        model = load_model('coin_predictor.h5')
        preds = model.predict(X_test, batch_size=2)
        preds_rewind = DataPrep.normalize_rewind(preds, price_matrix)
        GaiPlot.print_prediction_plot_compare(dates_matrix, preds_rewind)
        #plotlist = DataPrep.deserializer(preds, X_test, train_phase=True)

        # preds_rewind = DataPrep.normalize_rewind(preds, price_matrix)
        # GaiPlot.print_prediction_plot_compare(dates_matrix, preds_rewind)
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # fig.autofmt_xdate()
        # ax.plot(dates_matrix, preds_rewind)
        # ax.plot(dates_matrix, prices[29:len(prices)])
        # plt.show()

if __name__ == "__main__":
    RnnPredict.train()