import pandas as pd

from sqlalchemy import create_engine

engine = create_engine("postgresql://rea:rea@147.102.6.64:5555")

df = pd.read_csv("datasets/Energyefficiency_measures.csv", encoding='utf-16')

df = df[df['place'].notna()]
df.drop(columns=['Pasākums (Latvian)'], inplace=True)
df.to_csv('datasets/CSV/energy_efficiency_measures.csv', index=False, encoding='utf-16')
df.to_sql("energy_efficiency_measures", engine, index = False, if_exists = 'append')
