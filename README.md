# gai_trade
A collection of Crypto trading algorithms in the search of a reasonable trading strategy.

## LSTM Deep Learning
Check out the folder machine_learning, in the file rnn_predict_7.py are some explanations. For now it was more play around code. The predictions are actually quite accurate, but the LSTM network could in the end not say much more that "statistically its going up in the long term".

## Trying out ideas
The file bot.py is executing an algorithm with some ideas, which seem to work to a degree. The Fear and Greed Index is used for buying and it is sold when it was going up for a longer time. I cross checked with 4 currencies and a 2x over 3 years should be possible (amount of coins owned). That number is not satisfying for me yet as there is a high risk involved. Try it out for yourself :)
