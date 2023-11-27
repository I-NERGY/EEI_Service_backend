import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql://rea:rea@147.102.6.64:5555")

df = pd.read_csv("datasets/Buidlind_data_audits_1.csv", encoding='utf-16')
df2 = pd.read_csv("datasets/Buidlind_data_audits_2.csv", encoding='utf-16')





"""
CREATE TABLE investments
investments = result[['cadastre_number', 'total_renovation_cost', 'total_renovation_cost_per_m2']].copy()
investments.drop_duplicates(subset=['cadastre_number'], inplace=True)
print (investments['cadastre_number'].duplicated().any())
#investments.rename(columns={c:  '€' + c for c in result.values if '€' in c }, inplace=True)
#investments['total_renovation_cost'] = investments['total_renovation_cost'].map(lambda x: x.lstrip('+-').rstrip('€'))
#investments['total_renovation_cost_per_m2'].replace('€/','',regex=True,inplace=True)

investments.to_csv('datasets/CSV/investments.csv', index=False, encoding='utf-16')
investments.to_sql("investments", engine, index = False, if_exists = 'append')
"""




"""
CREATE TABLE ENVELOPE 

df = df[df['Sample data set'].notna()]
df['structure_heat_loss_coefficient'].fillna(0,inplace=True)
df = df.reset_index()

for i in range(len(df['structure_heat_loss_coefficient'])):
    if df['structure_heat_loss_coefficient'][i] == 0:
        df['structure_heat_loss_coefficient'][i] =   df['structure_heat_loss_coefficient'][i+1] + df['structure_heat_loss_coefficient'][i+2]     
#print(df['Structure heat loss coefficient'])



df.drop(df[df['Sample data set'] == 'Zone 1'].index, inplace=True)
df.drop(df[df['Sample data set'] == 'Zone 2'].index, inplace=True)
#df = df[df['Sample data set'].notna()]

df2.drop(df2[df2['Sample data set'] == 'Zone 1'].index, inplace=True)
df2.drop(df2[df2['Sample data set'] == 'Zone 2'].index, inplace=True)
df2 = df2[df2['Sample data set'].notna()]

frames = [df, df2]
result = pd.concat(frames)

result.drop_duplicates(subset=['Address'], inplace=True)
#result.drop(columns=['Address', 'Type of apartment building (building serie)', 'Sample data set', 'Kopējie siltuma ieguvumi', 'Kopējie siltuma ieguvumi.1'], inplace=True)


envelope = result[['cadastre_number', 'structure_heat_loss_coefficient', 'energy_consumption']].copy()
envelope.drop_duplicates(subset=['cadastre_number'], inplace=True)
#print (envelope['cadastre_number'].duplicated().any())
#envelope = envelope[envelope['Sample data set'].notna()]
envelope['cadastre_number'] = envelope['cadastre_number'].astype(int)
envelope['cadastre_number'] = envelope['cadastre_number'].astype(str)


envelope.to_csv('datasets/CSV/envelope.csv', index=False, encoding='utf-16')
envelope.to_sql("envelope", engine, index = False, if_exists = 'append')

"""




"""
CREATE TABLE ENERGY_CONSUMPTION
energy_consumption = result[['cadastre_number', 'heating_system', 'hot_water_system',	'heating_system_energy_consumption',	'hot_water_energy_consumption',	'total_energy_consumption',	'benefit_utilization_factor']].copy()
energy_consumption.drop_duplicates(subset=['cadastre_number'], inplace=True)
print (energy_consumption['cadastre_number'].duplicated().any())
energy_consumption.to_csv('datasets/CSV/energy_consumption.csv', index=False, encoding='utf-16')
energy_consumption.to_sql("energy_consumption", engine, index = False, if_exists = 'append')

"""




"""
CREATE TABLE CALCULATIONS
calculations = result[['cadastre_number','total_area', 'building_volume', 'temperature', 'heating_period', 'air_exchange']].copy()
calculations.drop_duplicates(subset=['cadastre_number'], inplace=True)
print (calculations['cadastre_number'].duplicated().any())
calculations.to_csv('datasets/CSV/calculations.csv', index=False, encoding='utf-16')
calculations.to_sql("calculations", engine, index = False, if_exists = 'append')

"""


"""
CREATE TABLE AUDITS
audits = result[['cadastre_number','serie', 'width', 'length', 'floors', 'volume', 'Avg_indoor_height', 'apartments', 'type_of_heating']].copy()
audits.drop_duplicates(subset=['cadastre_number'], inplace=True)
print (audits['cadastre_number'].duplicated().any())
audits.to_csv('datasets/CSV/audits.csv', index=False, encoding='utf-16')
audits.to_sql("audits", engine, index = False, if_exists = 'append')


"""



"""
CLEAR AUDIT DATA
df.drop(df[df['Sample data set'] == 'Zone 1'].index, inplace=True)
df.drop(df[df['Sample data set'] == 'Zone 2'].index, inplace=True)
df = df[df['Sample data set'].notna()]

df.to_csv('datasets/Building_data_audits_2_cleaned.csv', index=False, encoding='utf-16')
"""