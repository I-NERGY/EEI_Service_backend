import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql://<user>:<password>@<ip>:<port>")

df = pd.read_csv("datasets/Riga_DHS_sheet_1.csv", encoding='utf-16')
df2 = pd.read_csv("datasets/Riga_DHS_sheet_2.csv", encoding='utf-16')


#print(df[''])

frames = [df, df2]
result = pd.concat(frames)
#print (result['Address'].duplicated().any())
result.drop_duplicates(subset=['Address'], inplace=True)
result.drop(columns=['Address', 'Total Area', 'Useful area', 'mainusetype', '57'], inplace=True)

result.rename(columns={'cadastrenumber': 'cadastre_number'}, inplace=True)

result.rename(columns={c:  c + '0' for c in df.columns if '.202' in c and '.2021' not in c}, inplace=True)


result.to_csv('datasets/Riga_DHS_all.csv', index=False, encoding='utf-16')


result.to_sql("validation", engine, index = False, if_exists = 'append') 
 


"""
#DELETING NULL AND NOT MULTI APARTMENT BUILDINGS from sheet 1

df = df[df['Building type'].notna()]
df = df[~df["Building type"].str.contains("1122") == False]
print(df.head)
df.to_csv('datasets/Riga_DHS_sheet_1.csv', index=False, encoding='utf-16')
"""