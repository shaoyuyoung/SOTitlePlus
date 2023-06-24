"""
The cleaned corpus is divided in chronological order
"""
import pandas as pd
import numpy as np
from tqdm import *

lans = ['python', 'java', 'javascript', 'c#', 'php', 'html']  #

for lan in lans:
    from_file = f'./data/{lan}.csv'
    to_file = f'./data/{lan}/'

    df = pd.read_csv(from_file)

    df_train = df.iloc[0:4 * len(df) // 5]
    df_train.to_csv(to_file + '/train.csv', index=None, encoding='utf-8')

    df_valid = df.iloc[4 * len(df) // 5:9 * len(df) // 10]
    df_valid.to_csv(to_file + '/valid.csv', index=None, encoding='utf-8')

    df_test = df.iloc[9 * len(df) // 10:len(df) + 1]
    df_test.to_csv(to_file + '/test.csv', index=None, encoding='utf-8')
