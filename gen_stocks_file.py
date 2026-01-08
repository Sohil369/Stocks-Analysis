import pandas as pd

url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"# Fetch the list of all NSE Nifty 500 stocks
df = pd.read_csv(url)

df_top100 = df.head(100)# Get only the top 100 symbols

tickers = [t + ".NS" for t in df_top100["Symbol"]]

pd.DataFrame(tickers).to_excel("stocks.xlsx", index=False, header=False)

print(f"Created stocks.xlsx with {len(tickers)} top NSE tickers.")
