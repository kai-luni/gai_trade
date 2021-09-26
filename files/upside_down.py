file1 = open('files/BTC-USD.csv', 'r')
lines = file1.readlines()
file2 = open('files/btc_usd_yahoo_finance.csv', 'w')

for line in reversed(lines):
    file2.write(line)

file1.close()
file2.close()