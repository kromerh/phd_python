import pandas as pd

df = pd.DataFrame([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10],[0, 0, 0, 0, 0]],
                  columns=['A', 'B', 'C', 'D', 'E'])

print(df)

df_rev = df.iloc[::-1]

print(df_rev)
df = df.append(df_rev)
print(df)

df = df.shift(2)

print(df)
