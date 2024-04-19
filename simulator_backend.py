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


# sku_list_for_backend = [ 'MX06708A_61029529', '61013017' ]
sku_list_for_backend = ['MX06708A_61029529' ]
# Define start and end dates as strings
start_date = '29-04-2024'
end_date = '13-01-2025'

# total backend function 

def simulator_backend(sku_list_for_backend, start_date, end_date ):
    output_all_sku = pd.DataFrame()
    # for sku in front_end.SKU.unique():
    for sku in sku_list_for_backend :

        output = output_og[['feature'] + list(output_og.loc[:, start_date:end_date].columns)].copy()

        print("############### next sku ################")
        for date_col in output.columns[output.columns.get_loc(start_date):output.columns.get_loc(end_date)+1].values:
            temp_place_holder = 0
            units_sales_holder = 0

            print("############################# next date ###########################################")
            filtered_front_end = front_end[(front_end.SKU == sku) & (front_end.ws_date.dt.strftime('%d-%m-%Y') == date_col)]
            if len(filtered_front_end) > 0:
                if filtered_front_end.iloc[0]['rssp_flag'] == 0:
                    a = df_coeff[df_coeff['SKU'] == sku]['BASE_PRICE_LOG'].values[0]

                    b =  np.log(last_6m[last_6m['SKU'] == sku]['rssp_last_6m']).values[0] 

                    print('last 6 months rssp', last_6m[last_6m['SKU'] == sku]['rssp_last_6m'].values[0])
                    print('base price coeff', a, 'nplog_base', b)

                    print('the date column selected ', date_col)

                    base_price_impact = np.exp(a * b)


                    print('the base price impact is',base_price_impact)


                    print(sku,date_col,base_price_impact)
                    base_price_value = weekly_debased[ (weekly_debased['SKU'] == sku) & \
                                                       (weekly_debased['Updated_week_start_date']== date_col)]\
                                                    ['debase_price_sos'] * base_price_impact

                    print("base price incremental for the week ", date_col , base_price_value)

                    base_price_value = base_price_value.values[0]

                    # upadting sales holder
                    units_sales_holder = base_price_value



                    base_price_row_index = output[output['feature'] == 'base_price'].index
                    print(base_price_row_index)
                    output.loc[base_price_row_index, date_col] = base_price_value

                else:

                    a = df_coeff[df_coeff['SKU'] == sku]['BASE_PRICE_LOG'].values[0]
                    b = np.log((last_6m[last_6m['SKU'] == sku]['rssp_last_6m'] * (filtered_front_end.iloc[0]['rssp_change_%'] / 100 + 1)))

                    base_price_impact =  np.exp(a*b)
                    print('the base price impact is',base_price_impact)

                    print(sku,date_col,base_price_impact.values[0])
                    base_price_value = weekly_debased[ (weekly_debased['SKU'] == sku) & \
                                                       (weekly_debased['Updated_week_start_date']== date_col)]\
                                                    ['debase_price_sos'] * base_price_impact.values[0]

                    print("base price incremental for the week ", date_col , base_price_value)

                    base_price_value = base_price_value.values[0]

                    # upadting sales holder
                    units_sales_holder = base_price_value


                    base_price_row_index = output[output['feature'] == 'base_price'].index
                    output.loc[base_price_row_index, date_col] = base_price_value
            else:
                # if that particular data is not there 
                base_price_value = 1 
            #==================================================================================================            
            # calculating the sos incremental

            sos_coeff = df_coeff[df_coeff['SKU'] == sku]['SOS_VALUE'].values[0]
            sos_value = np.log(last_6m[last_6m['SKU'] == sku]['sos_last_6m']).values[0]

            sos_impact = np.exp(sos_coeff * sos_value)
            sos_incremental_final = sos_impact * base_price_value

            print('sos impact is ',sos_impact)


            # adding units sales holder
            units_sales_holder = sos_incremental_final


            print("sos incremental for the week ", date_col , sos_incremental_final)

            sos_row_index = output[output['feature'] == 'sos_value'].index
            output.loc[sos_row_index, date_col] = sos_incremental_final
            #=================================================================================
            # getting the final Baseline sales 

            total_final_baseline = sos_incremental_final
            total_bs_line_row_index = output[output['feature'] == 'Baseline Sales'].index
            output.loc[total_bs_line_row_index, date_col] = total_final_baseline

            temp_place_holder = total_final_baseline



            #==================================================================================================  

            disc_a = front_end[(front_end.SKU == sku) & (front_end.ws_date.dt.strftime('%d-%m-%Y') == date_col)] \
                                                                                            ['Discount %']

        #         if (disc_a.values[0] != np.nan):
            if (disc_a.values[0] != 0):
                print('discount from the front end', disc_a)
                discount_coeff = df_coeff[df_coeff['SKU'] == sku]['MANUFACTURER_TPR'].values[0]
                discount_value = np.log(1-(disc_a/100)).values[0]

                print('discount_coeff is ',discount_coeff , 'discount_value is ',  discount_value )



                discount_impact = np.exp(discount_value * discount_coeff)

                discount_incremental_final = sos_incremental_final * discount_impact

                print('discount impact is ',discount_impact)

                print("discount incremental for the week ", date_col , discount_incremental_final )

                discount_row_index = output[output['feature'] == 'Discount'].index
                output.loc[discount_row_index, date_col] = discount_incremental_final

                # adding units sales holder
                units_sales_holder = discount_incremental_final

                # adding incremental units of discount
                discount_inc_units = discount_incremental_final - temp_place_holder
                temp_place_holder = discount_incremental_final

                discount_inc_row_index = output[output['feature'] == 'Discount_inc_units'].index
                output.loc[discount_inc_row_index, date_col] = discount_inc_units


            else:
                # updating units sales holder if not there
                discount_incremental_final = units_sales_holder


                # adding incremental units of discount 
                discount_inc_units = 0
                discount_inc_row_index = output[output['feature'] == 'Discount_inc_units'].index
                output.loc[discount_inc_row_index, date_col] = discount_inc_units



            discount_row_index = output[output['feature'] == 'Discount'].index
            output.loc[discount_row_index, date_col] = discount_incremental_final




            #=============================================================================================

            # impact calculation for coe aretes 1

            if (front_end[(front_end.SKU == sku) & \
                          (front_end.ws_date.dt.strftime('%d-%m-%Y') == date_col)]['ARETES U OREJAS']== 'Yes').any():


                aretes_coeff = df_coeff[df_coeff['SKU'] == sku]['ARETES U OREJAS'].values[0]
                aretes_value = np.log(last_6m[last_6m['SKU'] == sku]['ARETES U OREJAS_last_6m'] + 1).values[0]

                aretes_impact = np.exp(aretes_coeff * aretes_value)
                aretes_incremental_final = discount_incremental_final * aretes_impact

                # adding incremental units aretes

                aretes_inc_units = aretes_incremental_final - temp_place_holder
                temp_place_holder = aretes_incremental_final

                # adding units sales holder
                units_sales_holder = aretes_incremental_final

                aretes_inc_row_index = output[output['feature'] == 'ARETES U OREJAS_inc_units'].index
                output.loc[aretes_inc_row_index, date_col] = aretes_inc_units


            else:
                # adding units sales holder if not there 
                aretes_incremental_final = units_sales_holder

                # adding incremental units aretes
                aretes_inc_units = 0 
                aretes_inc_row_index = output[output['feature'] == 'ARETES U OREJAS_inc_units'].index
                output.loc[aretes_inc_row_index, date_col] = aretes_inc_units


            aretes_row_index = output[output['feature'] == 'ARETES U OREJAS'].index
            output.loc[aretes_row_index, date_col] = aretes_incremental_final


            print('aretes impact is ',aretes_impact)
            print("aretes incremental for the week ", date_col , aretes_incremental_final)


            # ====================================================================================================


            # impact calculation for coe  cabecera 2

            if (front_end[(front_end.SKU == sku) & \
                          (front_end.ws_date.dt.strftime('%d-%m-%Y') == date_col)]['CABECERA']== 'Yes').any():


                cabecera_coeff = df_coeff[df_coeff['SKU'] == sku]['CABECERA'].values[0]
                cabecera_value = np.log(last_6m[last_6m['SKU'] == sku]['CABECERA_last_6m'] + 1).values[0]

                cabecera_impact = np.exp(cabecera_coeff * cabecera_value)
                cabecera_incremental_final = aretes_incremental_final * cabecera_impact

                # adding units sales holder
                units_sales_holder = cabecera_incremental_final


                # adding incremental units cabecera

                cabecera_inc_units = cabecera_incremental_final - temp_place_holder
                temp_place_holder = cabecera_incremental_final
                cabecera_inc_row_index = output[output['feature'] == 'CABECERA_inc_units'].index
                output.loc[cabecera_inc_row_index, date_col] = cabecera_inc_units

            else:
                # adding units sales holder if not there 
                cabecera_incremental_final = units_sales_holder

                # adding incremental units cabecera
                cabecera_inc_units = 0
                aretes_inc_row_index = output[output['feature'] == 'CABECERA_inc_units'].index
                output.loc[aretes_inc_row_index, date_col] = cabecera_inc_units



            print('cabaccera impact is ',cabecera_impact)

            print("cabecera incremental for the week ", date_col , cabecera_incremental_final )

            cabecera_row_index = output[output['feature'] == 'CABECERA'].index
            output.loc[cabecera_row_index, date_col] = cabecera_incremental_final

            # ====================================================================================================


            # impact calculation for coe  CANASTOS canastos 3

            if (front_end[(front_end.SKU == sku) & \
                          (front_end.ws_date.dt.strftime('%d-%m-%Y') == date_col)]['CANASTOS']== 'Yes').any():


                canastos_coeff = df_coeff[df_coeff['SKU'] == sku]['CANASTOS'].values[0]
                canastos_value = np.log(last_6m[last_6m['SKU'] == sku]['CANASTOS_last_6m'] + 1).values[0]



                canastos_impact = np.exp(canastos_coeff * canastos_value)
                canastos_incremental_final = cabecera_incremental_final * canastos_impact

                # adding units sales holder
                units_sales_holder = canastos_incremental_final

                # getting incremental units of canastos

                canastos_inc_units = canastos_incremental_final - temp_place_holder
                temp_place_holder = canastos_incremental_final
                canastos_inc_row_index = output[output['feature'] == 'CANASTOS_inc_units'].index
                output.loc[canastos_inc_row_index, date_col] = canastos_inc_units

                print('canatons incremental units',  canastos_inc_units)

            else:
                # adding units sales holder if not there 
                canastos_incremental_final = units_sales_holder

                #  getting incremental units of canastos

                canastos_inc_units = 0
                canastos_inc_row_index = output[output['feature'] == 'CANASTOS_inc_units'].index
                output.loc[canastos_inc_row_index, date_col] = canastos_inc_units


            print('canatos impact is ',canastos_impact)
            print("canastos incremental for the week ", date_col , canastos_incremental_final )


            canastos_row_index = output[output['feature'] == 'CANASTOS'].index
            output.loc[canastos_row_index, date_col] = canastos_incremental_final

            #===================================================================================================

            # impact calculation for coe  EXHIBIDOR COLGANTE 4

            if (front_end[(front_end.SKU == sku) & \
                          (front_end.ws_date.dt.strftime('%d-%m-%Y') == date_col)]['EXHIBIDOR COLGANTE']== 'Yes').any():


                exhibidor_coeff = df_coeff[df_coeff['SKU'] == sku]['EXHIBIDOR COLGANTE'].values[0]
                exhibidor_value = np.log(last_6m[last_6m['SKU'] == sku]['EXHIBIDOR COLGANTE_last_6m'] + 1).values[0]

                exhibidor_impact = np.exp(exhibidor_coeff * exhibidor_value)
                exhibidor_incremental_final = canastos_incremental_final * exhibidor_impact


                # adding units sales holder
                units_sales_holder = exhibidor_incremental_final

                # getting incremental units of exhibidor

                exhibidor_inc_units = exhibidor_incremental_final - temp_place_holder
                temp_place_holder = exhibidor_incremental_final
                exhibidor_inc_row_index = output[output['feature'] == 'EXHIBIDOR COLGANTE_inc_units'].index
                output.loc[exhibidor_inc_row_index, date_col] = exhibidor_inc_units

            else:
                # adding units sales holder if not there
                exhibidor_incremental_final = units_sales_holder

                # getting incremental units of exhibidor

                exhibidor_inc_units = 0
                exhibidor_inc_row_index = output[output['feature'] == 'EXHIBIDOR COLGANTE_inc_units'].index
                output.loc[exhibidor_inc_row_index, date_col] = exhibidor_inc_units


            print('exhibidor impact is ',exhibidor_impact)    

            print("exhibidor incremental for the week ", date_col , exhibidor_incremental_final )

            exhibidor_row_index = output[output['feature'] == 'EXHIBIDOR COLGANTE'].index
            output.loc[exhibidor_row_index, date_col] = exhibidor_incremental_final


            #===================================================================================================

            # impact calculation for coe   MINI GONDOLA 5

            if (front_end[(front_end.SKU == sku) & \
                          (front_end.ws_date.dt.strftime('%d-%m-%Y') == date_col)]['MINI GONDOLA']== 'Yes').any():


                mini_coeff = df_coeff[df_coeff['SKU'] == sku]['MINI GONDOLA'].values[0]
                mini_value = np.log(last_6m[last_6m['SKU'] == sku]['MINI GONDOLA_last_6m'] + 1).values[0]

                mini_impact = np.exp(mini_coeff * mini_value)
                mini_incremental_final = exhibidor_incremental_final * mini_impact

                # adding units sales holder
                units_sales_holder = mini_incremental_final


                # getting incremental units of mini

                mini_inc_units = mini_incremental_final - temp_place_holder
                temp_place_holder = mini_incremental_final
                mini_inc_row_index = output[output['feature'] == 'MINI GONDOLA_inc_units'].index
                output.loc[mini_inc_row_index, date_col] = mini_inc_units 

            else:
                # adding units sales holder if its not there
                mini_incremental_final = units_sales_holder

                # getting incremental units of mini

                mini_inc_units = 0
                mini_inc_row_index = output[output['feature'] == 'MINI GONDOLA_inc_units'].index
                output.loc[mini_inc_row_index, date_col] = mini_inc_units



            print("mini incremental for the week ", date_col , mini_incremental_final )

            print('mini impact is ',mini_impact)

            mini_row_index = output[output['feature'] == 'MINI GONDOLA'].index
            output.loc[mini_row_index, date_col] = mini_incremental_final

            #=====================================================================================

            # impact calculation for coe   OTRAS EXHIBICIONES 6

            if (front_end[(front_end.SKU == sku) & \
                          (front_end.ws_date.dt.strftime('%d-%m-%Y') == date_col)]['OTRAS EXHIBICIONES']== 'Yes').any():


                otras_coeff = df_coeff[df_coeff['SKU'] == sku]['OTRAS EXHIBICIONES'].values[0]
                otras_value = np.log(last_6m[last_6m['SKU'] == sku]['OTRAS EXHIBICIONES_last_6m'] + 1).values[0]

                print( 'otras_coeff', otras_coeff, 'otras_value' , otras_value )

                otras_impact = np.exp(otras_coeff * otras_value)
                otras_incremental_final = mini_incremental_final * otras_impact

                # adding units sales holder
                units_sales_holder = otras_incremental_final

                # getting incremental units of otras

                otras_inc_units = otras_incremental_final - temp_place_holder
                temp_place_holder = otras_incremental_final
                otras_inc_row_index = output[output['feature'] == 'OTRAS EXHIBICIONES_inc_units'].index
                output.loc[otras_inc_row_index, date_col] = otras_inc_units

            else:
                # adding units sales holder if its not there
                otras_incremental_final = units_sales_holder

                # getting incremental units of otras

                otras_inc_units = 0
                otras_inc_row_index = output[output['feature'] == 'OTRAS EXHIBICIONES_inc_units'].index
                output.loc[otras_inc_row_index, date_col] = otras_inc_units


            print("otras incremental for the week ", date_col , otras_incremental_final )

            print('otras impact is ',otras_impact)

            otras_row_index = output[output['feature'] == 'OTRAS EXHIBICIONES'].index
            output.loc[otras_row_index, date_col] = otras_incremental_final


            # ============================================================================================

             # impact calculation for coe   PALLET 7


            if (front_end[(front_end.SKU == sku) & \
                          (front_end.ws_date.dt.strftime('%d-%m-%Y') == date_col)]['PALLET']== 'Yes').any():


                pallet_coeff = df_coeff[df_coeff['SKU'] == sku]['PALLET'].values[0]
                pallet_value = np.log(last_6m[last_6m['SKU'] == sku]['PALLET_last_6m'] + 1).values[0]

                print( 'pallet coeff is ', pallet_coeff, 'pallet_value is ', pallet_value )

                pallet_impact = np.exp(pallet_coeff * pallet_value)

                print( 'pallet impact is  ',  pallet_impact)
                pallet_incremental_final = otras_incremental_final * pallet_impact

                # adding units sales holder
                units_sales_holder = pallet_incremental_final


                # getting incremental units of pallet

                pallet_inc_units = pallet_incremental_final - temp_place_holder
                temp_place_holder = pallet_incremental_final
                pallet_inc_row_index = output[output['feature'] == 'PALLET_inc_units'].index
                output.loc[pallet_inc_row_index, date_col] = pallet_inc_units

            else:
                # adding units sales holder if its not there
                pallet_incremental_final = units_sales_holder

                pallet_inc_units = 0
                pallet_inc_row_index = output[output['feature'] == 'PALLET_inc_units'].index
                output.loc[pallet_inc_row_index, date_col] = pallet_inc_units



            print('pallet impact is ',pallet_impact)
            print("pallet incremental for the week ", date_col , pallet_incremental_final )

            pallet_row_index = output[output['feature'] == 'PALLET'].index
            output.loc[pallet_row_index, date_col] = pallet_incremental_final


            # ============================================================================================

             # impact calculation for coe   RACK 8


            if (front_end[(front_end.SKU == sku) & \
                          (front_end.ws_date.dt.strftime('%d-%m-%Y') == date_col)]['RACK']== 'Yes').any():


                rack_coeff = df_coeff[df_coeff['SKU'] == sku]['RACK'].values[0]
                rack_value = np.log(last_6m[last_6m['SKU'] == sku]['RACK_last_6m'] + 1).values[0]

                rack_impact = np.exp(rack_coeff * rack_value)
                rack_incremental_final = pallet_incremental_final * rack_impact

                # adding units sales holder
                units_sales_holder = rack_incremental_final

                # getting incremental units of rack

                rack_inc_units = rack_incremental_final - temp_place_holder
                temp_place_holder = rack_incremental_final
                rack_inc_row_index = output[output['feature'] == 'RACK_inc_units'].index
                output.loc[rack_inc_row_index, date_col] = rack_inc_units

            else:
                # adding units sales holder if not there
                rack_incremental_final = units_sales_holder

                # getting incremental units of rack

                rack_inc_units = 0
                rack_inc_row_index = output[output['feature'] == 'RACK_inc_units'].index
                output.loc[rack_inc_row_index, date_col] = rack_inc_units

            print("rack incremental for the week ", date_col ,rack_incremental_final )

            print('rack impact is ',rack_impact)

            rack_row_index = output[output['feature'] == 'RACK'].index
            output.loc[rack_row_index, date_col] = rack_incremental_final


            # ============================================================================================

             # impact calculation for coe   TIRAS 9



            if (front_end[(front_end.SKU == sku) & \
                          (front_end.ws_date.dt.strftime('%d-%m-%Y') == date_col)]['TIRAS']== 'Yes').any():


                tiras_coeff = df_coeff[df_coeff['SKU'] == sku]['TIRAS'].values[0]
                tiras_value = np.log(last_6m[last_6m['SKU'] == sku]['TIRAS_last_6m'] + 1).values[0]

                tiras_impact = np.exp(tiras_coeff * tiras_value)
                tiras_incremental_final = rack_incremental_final * tiras_impact


                # adding units sales holder
                units_sales_holder = tiras_incremental_final


                # getting incremental units of tiras

                tiras_inc_units = tiras_incremental_final - temp_place_holder
                temp_place_holder = tiras_incremental_final
                tiras_inc_row_index = output[output['feature'] == 'TIRAS_inc_units'].index
                output.loc[tiras_inc_row_index, date_col] = tiras_inc_units

            else:
                # adding units sales holder if not there
                tiras_incremental_final = units_sales_holder

                # getting incremental units of tiras

                tiras_inc_units = 0
                tiras_inc_row_index = output[output['feature'] == 'TIRAS_inc_units'].index
                output.loc[tiras_inc_row_index, date_col] = tiras_inc_units




            print("tiras incremental for the week ", date_col ,tiras_incremental_final )

            print('tiras impact', tiras_impact)




            tiras_row_index = output[output['feature'] == 'TIRAS'].index
            output.loc[tiras_row_index, date_col] = tiras_incremental_final

            # ================================================================================================

            # impact calculation of vc 

            if (front_end[(front_end.SKU == sku) & \
                          (front_end.ws_date.dt.strftime('%d-%m-%Y') == date_col)]['Virtual Combos'].any()):


                vc_selected = front_end[(front_end.SKU == sku) & \
                          (front_end.ws_date.dt.strftime('%d-%m-%Y') == date_col)]['Virtual Combos'].values[0]

                # adding that virtual combo to the display output
                vc_display_index = output[output['feature'] == 'Virtual Combos'].index
                output.loc[vc_display_index, date_col] = vc_selected

                print('selected vc is ', vc_selected)


                vc_coeff = df_coeff[df_coeff['SKU'] == sku][vc_selected].values[0]
                vc_value = 1

                vc_impact = np.exp(vc_coeff*vc_value )
                vc_incremental_final  = tiras_incremental_final * vc_impact

                # adding units sales holder
                units_sales_holder = vc_incremental_final

                # getting  incremental units of virtual combo 
                vc_inc_units = vc_incremental_final - temp_place_holder
                temp_place_holder = vc_incremental_final
                vc_inc_row_index = output[output['feature'] == vc_selected+'_inc_units'].index
                output.loc[vc_inc_row_index, date_col] = vc_inc_units

                print( 'vc_incremental impact', vc_impact)


            else:
                # adding units sales holder if its not there 
                vc_incremental_final = units_sales_holder


                # adding that virtual combo to the display output
                vc_display_index = output[output['feature'] == 'Virtual Combos'].index
                output.loc[vc_display_index, date_col] = 'None'

                # getting  incremental units of virtual combo 
                vc_inc_units = 0
                vc_inc_row_index = output[output['feature'] == vc_selected+'_inc_units'].index
                output.loc[vc_inc_row_index, date_col] = vc_inc_units

            print("vc incremental for the week ", date_col ,vc_incremental_final )

            print('vc impact', vc_impact)

            vc_row_index = output[output['feature'] == vc_selected].index
            output.loc[vc_row_index, date_col] = vc_incremental_final

            # ====================================================================

            # updating unit sales
            units_row_index = output[output['feature'] == 'Unit Sales'].index
            output.loc[units_row_index, date_col] = vc_incremental_final

            # =========================================================================
            # calculating the final incremental 



            total_final_incremental = vc_incremental_final - total_final_baseline
            total_inc_line_row_index = output[output['feature'] == 'Incremental Sales'].index
            output.loc[total_inc_line_row_index, date_col] = total_final_incremental


            #=====================================================================================
            # revenue claculation and roi 
            filtered_front_end = front_end[(front_end.SKU == sku) & (front_end.ws_date.dt.strftime('%d-%m-%Y') == date_col)]
            if len(filtered_front_end) > 0:
                if filtered_front_end.iloc[0]['rssp_flag'] == 0:

                    unit_sales = vc_incremental_final
                    rssp =  last_6m[last_6m['SKU'] == sku]['rssp_last_6m'].values[0]
                    disc_a = front_end[(front_end.SKU == sku) & (front_end.ws_date.dt.strftime('%d-%m-%Y') == date_col)] \
                                                                                            ['Discount %'].values[0]
                    tpr = (1-(disc_a/100))


                    revenue = unit_sales* (rssp * tpr )

                    print( 'the revenue is ',revenue )

                    revenue_row_index = output[output['feature'] == 'Revenue'].index
                    output.loc[revenue_row_index, date_col] = revenue

                    # roi calculation
                    investment = vc_incremental_final * rssp * disc_a/100
                    roi = (revenue/investment)
                    print( 'the roi is ', roi )

                    roi_row_index = output[output['feature'] == 'ROI'].index
                    output.loc[roi_row_index, date_col] = roi



                else:

                    unit_sales = vc_incremental_final
                    rssp = last_6m[last_6m['SKU'] == sku]['rssp_last_6m'] * (filtered_front_end.iloc[0]['rssp_change_%'] / 100 + 1)
                    rssp = rssp.values[0]
                    disc_a = front_end[(front_end.SKU == sku) & (front_end.ws_date.dt.strftime('%d-%m-%Y') == date_col)] \
                                                                                            ['Discount %']
                    tpr = (1-(disc_a/100)).values[0]

                    revenue = unit_sales* (rssp * tpr )

                    print( 'the revenue is ',revenue )

                    revenue_row_index = output[output['feature'] == 'Revenue'].index
                    output.loc[revenue_row_index, date_col] = revenue

                    # roi calculation
                    investment = vc_incremental_final * rssp * disc_a.values[0]/100
                    roi = (revenue/investment)

                    print( 'the roi is ', roi )

                    roi_row_index = output[output['feature'] == 'ROI'].index
                    output.loc[roi_row_index, date_col] = roi



            else:
                revenue = 1 # if that particular data is not there
                roi = 1

             #=====================================================================================

        output.insert(0, 'SKU', sku)
        output.insert(1, 'Description', front_end[front_end['SKU'] == sku]['DESCRIPTION'].values[0])

        output_all_sku = pd.concat([output_all_sku, output], ignore_index=True)
    
    return output_all_sku      

g = simulator_backend(sku_list_for_backend,start_date, end_date )


