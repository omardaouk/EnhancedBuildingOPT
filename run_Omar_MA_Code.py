"""
Created on Wed July 13 14:22:00 2019

@author: Omar Daouk
"""
from __future__ import division
import Omar_MA_Code as opti_model


from openpyxl import load_workbook
from openpyxl.styles import (Font, Border, Side, Alignment)
from openpyxl import Workbook
from openpyxl.chart import (AreaChart, ScatterChart, Reference, Series)
import datetime
import functions_Omar_MA as function

"""
3 functions to run: 

co2_vs_storage where the capacity of the battery, TES and both can be varied
co2_vs_envelope where the envelope upgrades are varied
multi_objective_optimization where the multiple objective optimization is run

"""
if __name__ == "__main__":
    # Choose which function to run and comment the rest out
    #x = "co2_vs_storage"
    #x = "co2_vs_envelope"
    x = "multi_objective_optimization"

    if x == "co2_vs_storage":

        # Name of the output excel file for the results
        name = "Co2 vs TES STC with area STC = 10 CO2(t) 2017, LCA , Self Discharge) 4 type weeks"

        # Choose which storage system to vary and comment the rest out
        variable = "Battery"
        #variable = "TES"
        #variable = "Battery & TES"

        if variable == "Battery":
            x_axis = "Battery capacity in kwh"
        elif variable == "TES":
            x_axis = "TES capacity in m3"
        elif variable == "Battery & TES":
            x_axis = "Total Storage Capacity (kwh/m3)"

        y_axis_1 = " Emissions in tCO2 per year"
        y_axis_2 = "costs in 1000 Eur. Per year"
        print("Running test Scenario")
        if x_axis == "Battery capacity in kwh":
            upper_limit = 30  # kwh and 5000 if testing seasonal Storage
        elif x_axis == "TES capacity in m3":
            upper_limit = 2  # m3
        elif x_axis == "Total Storage Capacity (kwh/m3)":
            upper_limit = 32  # m3

        # The upper limits are also restricted by the ones in the main script ... as long as
        # these are within them it is okay ... if these are increased the ones in the main script must be increased
        # accordingly

        function.run_CO2_vs_Storage_typ_wochen(number_simulations=20, name_of_file=name, name_of_x=x_axis,
                                               name_of_y1=y_axis_1, name_of_y2=y_axis_2,
                                               storage_max=upper_limit, Variable=variable,
                                               enev_restrictions=False, pv_scenario=False, status_quo_with_storage=True,
                                               )

    elif x == "co2_vs_envelope":

        # Name of the output excel file for the results
        name = "CO2 vs Envelop( CO2(t) 2050 LCA , Self Discharge) 4 type weeks"

        envelope_matrix = {"wall": 0, "roof": 0, "window": 0}  # Start off with Status Quo

        bes_matrix = {"CHP": 0, "HP": 0, "STC": 0, "TES0": 1, "TES": 0, "Boiler0": 1, "Boiler": 0, "EH": 0, "STC": 0,  "Battery": 0, "PV": 0} # Start off with Status Quo

        print("########################")
        print("########################")
        print("Running_scenario", )
        print("########################")
        print("########################")

        function.run_CO2_vs_Envelope_typ_wochen(name_of_file=name, envelope_details=envelope_matrix, bes_details=bes_matrix)


    elif x == "multi_objective_optimization":
        # Possible scenarios:
        # pv    enev
        # True  irrelevant
        # False True
        # False False


        # PV scenario
        #print("Running PV scenario")
        #run_multi_obj(number_simulations=1,
        #             enev_restrictions=True,
        #             pv_scenario=True)

        # EnEV scenario
        #print("Running EnEV scenario")
        #run_multi_obj(number_simulations=1,
        #            enev_restrictions=True,
        #            pv_scenario=False)

        # Without restrictions
        print("Running free scenario")

        # Name of the output excel file for the results
        name_of_file = "Free Run CO2(t) 2050 With 0 weight CO2(t)"
        function.run_multi_obj(name_of_output=name_of_file, number_simulations=8,
                      enev_restrictions=False,
                      pv_scenario=False)
