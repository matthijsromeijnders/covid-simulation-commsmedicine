import configparser
import pandas as pd
import numpy as np

config = configparser.ConfigParser()
config.read('config.ini')

data_gem_inw = pd.read_csv(config['PATHS']['RAWDATA_GEM_INW'], delimiter=';', encoding='latin-1')
data_gem_alfa = pd.read_csv(config['PATHS']['RAWDATA_GEM_ALFA'], delimiter=';', encoding='latin-1')

data_gem_inw.dropna(subset='TotaleBevolking_1', inplace=True)

new_data = pd.merge(data_gem_inw, data_gem_alfa[['GemeentecodeGM','Gemeentenaam']], on='GemeentecodeGM')
new_data['GemeentecodeGM'] = new_data['GemeentecodeGM'].apply(lambda gmId: int(gmId.lstrip('GM')))
new_data.rename(columns={"GemeentecodeGM": "Gemeentecode"}, inplace=True)

new_data.to_csv('../Data/Gem_inwoners_totaal_2018.csv', sep=';', encoding='utf-8')
