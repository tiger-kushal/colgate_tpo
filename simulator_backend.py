import pandas as pd
import numpy as np
import datetime
from distutils.command.config import config
from tqdm.auto import tqdm
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 1000)
pd.set_option("display.max_rows", 10000)
import re

pd.set_option('display.float_format', lambda x: '%.3f' % x)

weekly_debased = pd.read_csv("debased_weekly_sales_filled_missing_dates_1225_till_v2.csv").drop('Unnamed: 0', axis =1 )
last_6m = pd.read_excel('all_last_6months_kml.xlsx')
df_coeff = pd.read_excel("BA_Simulator.xlsm",sheet_name = "Coeff_Summary")
front_end = pd.read_excel('front_end_filtered_1225_till.xlsx')
output_og = pd.read_csv('output_og.csv').drop('Unnamed: 0', axis =1 )

weekly_debased['SKU'] = weekly_debased['SKU'].astype('str')
last_6m['SKU'] = last_6m['SKU'].astype('str')
df_coeff['SKU'] = df_coeff['SKU'].astype('str')
front_end['SKU'] = front_end['SKU'].astype('str')


