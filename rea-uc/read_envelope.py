import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql://rea:rea@147.102.6.64:5555")

df = pd.read_csv("datasets/CSV/Building_envelope_audits_1_cleaned.csv", encoding='utf-16')
df2 = pd.read_csv("datasets/CSV/Building_envelope_audits_2_cleaned.csv", encoding='utf-16')


#print(df[''])

frames = [df, df2]
result = pd.concat(frames)

result.to_csv('datasets/CSV/envelope_components.csv', index=False, encoding='utf-16')

result.to_sql("envelope_components", engine, index = False, if_exists = 'append') 

"""
MAKE ENVELOPE 2 INTO READABLE FORM 

def copy_columns(row):def copy_columns(row):
    
    if pd.isna(row['Enclosing structure']):
        row['total_structure_heat_loss_coefficient'] = row['Structure heat loss coefficient']
        row['total_energy_consumption_x'] = row['Energy consumption = 10 x 9 x number of heating days x hours']
        row['total_area_x'] = row['Area']
    return row

def sum_column_and_replace(result, target_column):
    total_sum = 0
    for index, row in result.iterrows():
        
        if row[target_column] == "Total: ":
            
            result.at[index, target_column] = total_sum
            #print(result.at[index, target_column])
            total_sum = 0
        else:
            total_sum += float(row[target_column])
    return result

df = pd.read_csv("datasets/Buidlind_envelope_audits_1.csv", encoding='utf-16')

audits = pd.read_csv("datasets/CSV/Building_data_audits_1_cleaned.csv", encoding='utf-16')


result = pd.merge(df, audits, on='Sample_data_set', how='left')

result = result[~result['Sample_data_set'].str.contains('ZONE', na=False)]
result = result[~result['Sample_data_set'].str.contains('Zone', na=False)]
result['cadastre_number'] = result['cadastre_number'].fillna(method='ffill')
result.drop(columns=['Address','Sample_data_set'], inplace=True)
#result = result[result['Structure heat loss coefficient'].notna()]

result.dropna(how='any',subset=['Structure heat loss coefficient'], inplace=True)
#result['Structure heat loss coefficient'] = pd.to_numeric(result['Structure heat loss coefficient'], errors='coerce')
#result.reset_index(drop=True)
sum_column_and_replace(result,2)
result = result.apply(copy_columns, axis=1)

result['total_structure_heat_loss_coefficient'] = result['total_structure_heat_loss_coefficient'].fillna(method='bfill')
result['total_energy_consumption_x'] = result['total_energy_consumption_x'].fillna(method='bfill')
result['total_area_x'] = result['total_area_x'].fillna(method='bfill')
result.to_csv('datasets/Building_envelope_audits_1_cleaned.csv', index=False,  encoding='utf-16')

    if row['Sample_data_set'] == 'Total Zone 1&2':
        row['total_structure_heat_loss_coefficient'] = row['structure_heat_loss_coefficient_x']
        row['total_energy_consumption_x'] = row['energy_consumption_x']
        row['total_area_x'] = row['area']
    return row


df = pd.read_csv("datasets/Buidlind_envelope_audits_2.csv", encoding='utf-16')

audits = pd.read_csv("datasets/CSV/Building_data_audits_2_cleaned.csv", encoding='utf-16')

result = pd.merge(df, audits, on='Sample_data_set', how='left')
result['cadastre_number'] = result['cadastre_number'].fillna(method='ffill')

result.drop(result[result['Sample_data_set'] == 'ZONE 1&2'].index, inplace=True)
result = result[~result['Sample_data_set'].str.contains('BUILDING', na=False)]
result = result.apply(copy_columns, axis=1)
result['total_structure_heat_loss_coefficient'] = result['total_structure_heat_loss_coefficient'].fillna(method='bfill')
result['total_energy_consumption_x'] = result['total_energy_consumption_x'].fillna(method='bfill')
result['total_area_x'] = result['total_area_x'].fillna(method='bfill')
result.drop(columns=['Address','Sample_data_set'], inplace=True)
result = result[result['Enclosing structure'].notna()]


result.to_csv('datasets/Building_envelope_audits_2_cleaned.csv', index=False, encoding='utf-16')
"""


"""
MAKE ENVELOPE 1 INTO READABLE FORM


def copy_columns(row):
    
    if pd.isna(row['Enclosing structure']):
        row['total_structure_heat_loss_coefficient'] = row['Structure heat loss coefficient']
        row['total_energy_consumption_x'] = row['Energy consumption = 10 x 9 x number of heating days x hours']
        row['total_area_x'] = row['Area']
    return row

def sum_column_and_replace(result, target_column):
    total_sum = 0
    for index, row in result.iterrows():
        
        if row[target_column] == "Total: ":
            
            result.at[index, target_column] = total_sum
            #print(result.at[index, target_column])
            total_sum = 0
        else:
            total_sum += float(row[target_column])
    return result

df = pd.read_csv("datasets/Buidlind_envelope_audits_1.csv", encoding='utf-16')

audits = pd.read_csv("datasets/CSV/Building_data_audits_1_cleaned.csv", encoding='utf-16')


result = pd.merge(df, audits, on='Sample_data_set', how='left')

result = result[~result['Sample_data_set'].str.contains('ZONE', na=False)]
result = result[~result['Sample_data_set'].str.contains('Zone', na=False)]
result['cadastre_number'] = result['cadastre_number'].fillna(method='ffill')
result.drop(columns=['Address','Sample_data_set'], inplace=True)
#result = result[result['Structure heat loss coefficient'].notna()]

result.dropna(how='any',subset=['Structure heat loss coefficient'], inplace=True)
#result['Structure heat loss coefficient'] = pd.to_numeric(result['Structure heat loss coefficient'], errors='coerce')
#result.reset_index(drop=True)
sum_column_and_replace(result,2)
result = result.apply(copy_columns, axis=1)

result['total_structure_heat_loss_coefficient'] = result['total_structure_heat_loss_coefficient'].fillna(method='bfill')
result['total_energy_consumption_x'] = result['total_energy_consumption_x'].fillna(method='bfill')
result['total_area_x'] = result['total_area_x'].fillna(method='bfill')
result.to_csv('datasets/Building_envelope_audits_1_cleaned.csv', index=False,  encoding='utf-16')








SECOND ATTEMPT USING THE ALREADY CLEAND CSV AGAIN



def copy_columns(row):
    
    if pd.isna(row['Enclosing structure']):
        row['total_structure_heat_loss_coefficient'] = row['Structure heat loss coefficient']
        row['total_energy_consumption_x'] = row['Energy consumption = 10 x 9 x number of heating days x hours']
        row['total_area_x'] = row['Area']
    return row

def replace_totals(df, column):
    sum_ = 0.0
    for i, value in df[column].iteritems():
        if value == 'Total: ':
            df.loc[i, column] = sum_
            sum_ = 0
        else:
            sum_ += float(value)
df2 = pd.read_csv("datasets/Building_envelope_audits_1_cleaned.csv", encoding='utf-16')

replace_totals(df2,'Structure heat loss coefficient')
df2 = df2.apply(copy_columns, axis=1)

df2['total_structure_heat_loss_coefficient'] = df2['total_structure_heat_loss_coefficient'].fillna(method='bfill')
df2['total_energy_consumption_x'] = df2['total_energy_consumption_x'].fillna(method='bfill')
df2['total_area_x'] = df2['total_area_x'].fillna(method='bfill')
df2.dropna(subset=['Enclosing structure'], inplace=True)

df2.to_csv('datasets/Building_envelope_audits_1_cleaned.csv', index=False,  encoding='utf-16')


"""