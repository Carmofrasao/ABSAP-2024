import pandas as pd

data = pd.read_csv('../dataset/task2_test.csv', sep=';', index_col=0)

for i in range(len(data)):
    print(f'{data.iloc[i]["texto"]} --- {data.iloc[i]["aspect"]}')
    print()
