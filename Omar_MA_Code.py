#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed July 13 14:22:00 2019

@author: Omar Daouk
"""

import gurobipy as gp
import numpy as np
import sun
import input_data.components as components
import input_data.building as building
import pickle
from openpyxl import load_workbook  # My Addition
from Clustering import calculate_clusters_weeks
from openpyxl import Workbook
from Clustering import error_vs_number_of_clusters


def optimize_MA(max_emissions, options):
    # Constants
    direction = ("south", "west", "north", "east", "roof", "floor")
    direction2 = ("wall", "roof", "floor")
    direction3 = ("south", "west", "north", "east")
    direction4 = ("roof", "floor")

    # Time relevant parameters
    timesteps = 24  # Time steps per typical day
    weight_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # Days per month
    number_days = len(weight_days)  # Number of typical days per year
    # monthsCum0 = np.array([0] + weight_days[0:number_days - 1])
    # monthsCum0 = monthsCum0.cumsum()
    dt = 1  # Time step width in hours

    # 5R1C ISO 13790 Parameter
    # Constants for calculation of A_m, dependent of building class
    # (DIN EN ISO 13790, section 12.3.1.2, page 81, table 12)
    f_class = {}
    f_class["Am"] = [2.5, 2.5, 2.5, 3.0, 3.5]  # m²
    # specific heat transfer coefficient
    # (DIN EN ISO 13790, section 7.2.2.2, page 35)
    h_is = 3.45  # [W/(m²K)]
    # non-dimensional relation between the area of all indoor surfaces and the
    # effective floor area A["f"]
    # (DIN EN ISO 13790, section 7.2.2.2, page 36)
    lambda_at = 4.5
    # specific heat transfer coefficient
    # (DIN EN ISO 13790, section 12.2.2, page 79)
    h_ms = 9.1  # [W/(m²K)]
    ###
    ###
    # building parameters
    # South, West, North, East, Roof, PV/STC
    beta = building.beta  # slope angle
    gamma = building.gamma  # surface azimuth angle

    # Building geometry
    A = building.A  # [m2]
    V = building.V  # [m3] (A_f*heightWalls)

    # Form factor for radiation between the element and the sky
    # (DIN EN ISO 13790, section 11.4.6, page 73)
    F_r = building.F_r

    # Internal gains in W
    phi_int = building.phi_int

    # Power generation / consumption in kW (positive, device dependent)
    power = building.power

    # Temperature bounds and ventilation input
    T_set_min = building.T_set_min  # °C
    ventilationRate = building.ventilationRate  # [airchanges per hour]

    # Approximated room temperature in °C
    T_i_appr = building.T_i_appr
    ###
    ###
    # Data depending on location (Aachen)
    ####################################
    # Missing: T_ne, T_me, f_g1, G_w, cp_air, rho_air
    weatherData = np.loadtxt("input_data/weather.txt", skiprows=2, delimiter="\t")
    weatherData = weatherData[0:8760, :]
    location = (49.5, 8.5)
    altitude = 0  # m, section 5.2.1.6.1, page 16
    timeZone = 1
    SunDirect = weatherData[:, 14]
    SunDiffuse = weatherData[:, 15]
    T_e_raw = weatherData[:, 9]  # Dry-Bulb Temperature
    albedo = 0.2  # Ground reflectance
    # Mean difference between outdoor temperature and the apparent sky-temperature
    # (DIN EN ISO 13790, section 11.4.6,  page 73)
    Delta_theta_er = 11  # [K]

    # Computation of design heat load
    T_ne = -23.33  # [°C] outside design temperature
    T_me = 10.08  # [°C] outside average temperature
    f_g1 = 1.45
    # Reduction factor
    f_g2 = (T_set_min - T_me) / (T_set_min - T_ne)
    G_w = 1.0  # influence of ground water neglected
    c_p_air = 1000.0  # [J/kgK]
    rho_air = 0.986  # [kg/m3]

    # Outdoor conditions (Sun radiation on surfaces and temperature)
    SunRad = sun.getSolarGains(0, 3600, 8760,
                               timeZone=timeZone,
                               location=location,
                               altitude=altitude,
                               beta=beta,
                               gamma=gamma,
                               beam=SunDirect,
                               diffuse=SunDiffuse,
                               albedo=albedo)

    SunRad_weeks = []

    """
    The next part is dedicated to getting the 8736 values we want from Sun_rad and splitting it into 6 different parts
    so that it can go into the clustering ( so instead of an array of 6 arrays each having 8760 elements
    we would get 6 independent arrays of 8736 elements)
    Similar for T_E
    """

    for i in range(SunRad.shape[0]):
        temporary = []
        for j in range(SunRad.shape[1] - 24):
            temporary.append(SunRad[i][j])
        SunRad_weeks.append(temporary)
    SunRad_weeks = np.array(SunRad_weeks)
    T_e_raw_weeks = []
    for i in range(8736):
        T_e_raw_weeks.append(T_e_raw[i])
    T_e_raw_weeks = np.array(T_e_raw_weeks)

    """
    Depending on the year that we want to investigate load CO2(t) of that year as CO2_t
    So comment the others out and keep the one we want
    """

    CO2_t = np.loadtxt("CO2(t)_2017_weeks.txt")  # Average = 0.558873449
    # CO2_t = np.loadtxt("CO2(t)_2030_weeks.txt")  # Average = 0.482000029
    # CO2_t = np.loadtxt("CO2(t)_2050_weeks.txt")  # Average = 0.386809367
    raw_inputs = {}

    raw_inputs["CO2(t)"] = CO2_t
    raw_inputs["dhw"] = np.loadtxt("input_data/demand_domestic_hot_water 8736" + ".txt")
    raw_inputs["electricity"] = np.loadtxt("electricity_sfh_4_medium 8736.txt")
    raw_inputs["chp_rem"] = np.loadtxt("CHP_Remuneration 8736.txt")
    raw_inputs["sun_rad_0"] = SunRad_weeks[0]
    raw_inputs["sun_rad_1"] = SunRad_weeks[1]
    raw_inputs["sun_rad_2"] = SunRad_weeks[2]
    raw_inputs["sun_rad_3"] = SunRad_weeks[3]
    raw_inputs["sun_rad_4"] = SunRad_weeks[4]
    raw_inputs["sun_rad_5"] = SunRad_weeks[5]
    raw_inputs["T_e_raw"] = T_e_raw_weeks

    inputs_clustering = np.array([raw_inputs["CO2(t)"],
                                  raw_inputs["dhw"],
                                  raw_inputs["electricity"],
                                  raw_inputs["sun_rad_0"],
                                  raw_inputs["sun_rad_1"],
                                  raw_inputs["sun_rad_2"],
                                  raw_inputs["sun_rad_3"],
                                  raw_inputs["sun_rad_4"],
                                  raw_inputs["sun_rad_5"],
                                  raw_inputs["T_e_raw"],
                                  ])

    '''The error_vs_number_of_clusters function runs the clustering from 0 to the number_typwochen
    And plots the "distance" graph showing the difference between the clustered input and the original input
    '''
    do = False  # This should be set to true if we want to run the function that shows the error vs number of clusters
    if do == True:
        number_typwochen = 52
        error_vs_number_of_clusters(inputs_clustering, number_typwochen)

    """
        number_clusters dictates how many type periods will be used to represent the whole year ... in this case larger
        4 periods was the best option ... so 4 representative weeks
    """

    number_clusters = 4

    """
    The weight to assign for each input must also be specified before initiating the clustering.
    For that a detailed correlation analysis between the factors is needed.
    In this case however, the inputs that behave similarly share 1 weight unit between them ... while the others
    that are not similar to other inputs are assigned 1 weight unit.
    The CO2(t) factor in this case is 1 ... it needs to be set to 0 if to compare between multiple years
    """

    weights_of_input = [1, 1, 1, 1/2, 1/2, 1/2, 1, 1, 1/2, 1]
    # Sun Rad 0 and 5 are related ... when one increases or decreases the other does that as well
    # Sunrad 1 and Sunrad 2 are very similar ... almost identical ... and thus also vary similarly
    # This is why Sund Rad 0 and 5 are both 1/2 to give a full 1 weight together ... and the same goes for Sunrad 1 & 2
    (inputs, nc, z, obj) = calculate_clusters_weeks(inputs_clustering, number_clusters, norm=2, mip_gap=0.0,
                                                    weights=weights_of_input)

    clustered = {}
    clustered["representative_days"] = {}
    clustered["representative_days"]["CO2(t)"] = inputs[0]
    clustered["representative_days"]["dhw"] = inputs[1]
    clustered["representative_days"]["electricity"] = inputs[2]
    clustered["representative_days"]["sun_rad_0"] = inputs[3]
    clustered["representative_days"]["sun_rad_1"] = inputs[4]
    clustered["representative_days"]["sun_rad_2"] = inputs[5]
    clustered["representative_days"]["sun_rad_3"] = inputs[6]
    clustered["representative_days"]["sun_rad_4"] = inputs[7]
    clustered["representative_days"]["sun_rad_5"] = inputs[8]
    clustered["representative_days"]["T_e_raw"] = inputs[9]
    clustered["weights"] = nc
    clustered["z"] = z
    clustered["weights_of_input"] = weights_of_input
    clustered["obj"] = obj

    number_of_representative_periods = clustered["representative_days"]["T_e_raw"].shape[0]
    length_of_cluster = clustered["representative_days"]["T_e_raw"].shape[1]

    T_e = np.zeros((number_of_representative_periods, length_of_cluster))
    SunRad_part = np.zeros((6, number_of_representative_periods, length_of_cluster))

    # In this next part we combine the sun_rad sub arrays (that are now clustered ) into 1 array
    SunRad = np.array([clustered["representative_days"]["sun_rad_0"],
                       clustered["representative_days"]["sun_rad_1"],
                       clustered["representative_days"]["sun_rad_2"],
                       clustered["representative_days"]["sun_rad_3"],
                       clustered["representative_days"]["sun_rad_4"],
                       clustered["representative_days"]["sun_rad_5"],
                       ])

    for d in range(number_of_representative_periods):
        for t in range(length_of_cluster):
            T_e[d, t] = clustered["representative_days"]["T_e_raw"][d][t]
            SunRad_part[:, d, t] = SunRad[:, d, t] / 1000  # kW

    # Building envelope components
    ##############################
    t_life = components.t_life  # Economic lifetime of different technologies [3]
    U = components.U  # thermal transmittance [kW/(m²*K)]
    epsilon = components.epsilon  # radiation emission coefficient [-]
    R_se = components.R_se  # surface resistance [m²*K/kW]
    alpha_Sc = components.alpha_Sc  # adsorption coefficient [-]
    kappa = components.kappa  # specific heat storage capacity [kWh/(K*m²]
    g_gl = components.g_gl  # overall energy transmittance of glazings
    inv = components.inv  # specific investion costs in [€/m²]

    # ratio of window-frame
    # (DIN EN ISO 13790, section 11.4.5, page 73)
    F_F = 0

    # thermal radiation transfer
    # [kW/(m²*K)] DIN EN ISO 13790, section 11.4.6, page 73
    h_r_factor = 5.0  # W / (m²K)
    h_r = {}
    h_r["opaque", "wall"] = h_r_factor * np.array(epsilon["opaque", "wall"])
    h_r["opaque", "roof"] = h_r_factor * np.array(epsilon["opaque", "roof"])
    h_r["opaque", "floor"] = h_r_factor * np.array(epsilon["opaque", "floor"])
    h_r["window"] = h_r_factor * np.array(epsilon["window"])

    # thermal transmittance coefficient H_ve [W/K]
    # (DIN EN ISO 13790, section 9.3.1, equation 21, page 49)
    H_ve = rho_air * c_p_air * ventilationRate * V / 3600

    # thermal transmittance coefficient H_tr_is [W/K]
    # (DIN EN ISO 13790, section 7.2.2.2, equation 9, page 35)
    A_tot = lambda_at * A["f"]
    H_tr_is = h_is * A_tot

    # heat flow phi_ia [W]
    # (DIN EN ISO 13790, section C2, page 110, eq. C.1)
    phi_ia = 0.5 * phi_int

    # matching coefficient for thermal transmittance coefficient if temperature is
    # unequal to T_e, otherwise = 1
    # Assumption: Constant annual heat flow through ground
    # (ISO 13370 A.5 p 25 eq. A8)
    T_e_mon = np.mean(T_e, axis=1)  # monthly mean outside temperature
    b_floor = []  # ground p.44 ISO 13790

    T_i_year = 22.917  # annual mean indoor temperature
    T_e_year = 9.71  # annual mean outside temperature

    """In order to avoide also adding T_i_appr[i] to the clustering which will increase the clustering error
    for the important things like CO2(t) (all for a value that has 1 value for each quarter of the year) 
    the following will be done:

    In Buildings.py T_i_appr is 20 for [Jan Feb March Apr] 27 for [May June July Aug Sept] and 20 for [Oct Nov Dec]

    AND VERY IMPORTANT : since the clustering takes the weeks that exist (uses kmedoids) instead of Building a week
    AND VERY IMPORTANT :  since the clustering function uses the weeks in a way that it takes the first week of the year as it is
    then the 2nd ... then 3rd and so on ... instead of taking any 7 days in a row ( it does Jan 1 - Jan7 as a week and
    not for example Jan 3- Jan 10 ) we can do the following :

    go week by week starting the 1st week of the year and assign the appropriate  value of T_i_appr

    For that i will check out which of the weeks of the year are fully within a singular month and which are not 
    (so which one start in month x and jump into month x+1)

    The weeks that are fully contained within a month will simply take the value of T_i_appr that corresponds to 
    that month.

    The other weeks are split into 2 groups :

    1- Those jump from month to another but are still within the 
    month groups of Building.py ( [Jan Feb March Apr] , [May June July Aug Sept], [Oct Nov Dec]) .. for these the 
    value of T_i_appr will also simply be the one that corresponds to the whole group

    2- Those that jump from a month to another and cross the groups while doing that ... in that case some of the 
    days of the week (the first days of the week (day 1 up to maybe day 6)) would have T_i_appr of the 1st group of 
    months they belong to .... while the rest (the last days of the week (maybe from day 2 up to day 7 )) 
    would have T_i_appr of the 2nd group of months they belong to ... in that case what i will do is 
    either use the weighted average for T_i_appr (more accurate) ... or let the majaurity decide 
    (if 4 days belong to group 1 then the value of group 1) (less accurate) ... either or depending on the way it is 
    later on used in the code

    THE MAIN THING TO DO AFTER THAT IS LOOK AT THE WEEK THAT REPRESENTS EACH CLUSTER
    and then check the above steps to decide what the value of T_i_appr for this week is 

    Even if we did not weigh T_i_appr in the clustering to choose weeks that
    are similar to each other ... from the dhw and electricity and T_e values .. the grouping of Cold and Hot days
    should already be done correctly ( Heating period from October until May  would be correctly grouped since heating 
    the heating period is directly affected by / directly affects the dhw and other stated factors) 
    """

    weeks_start_in_hours = []
    weeks_end_in_hours = []
    b = 1
    bb = 168
    for i in range(1, 8737, 7 * 24):
        weeks_start_in_hours.append(b)
        weeks_end_in_hours.append(bb)
        b += 7 * 24
        bb += 7 * 24
    weight_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # Days per month
    months_start_in_hours = []
    bb = 0
    for i in weight_days:
        months_start_in_hours.append(
            bb + 1)  # I want to switch not on the last hour of the months but on the first of the next
        # ... this makes sense the way i used this list in the following loop
        bb += i * 24
    exceptions = []

    for i in range(len(weeks_start_in_hours)):
        for j in range(len(
                months_start_in_hours)):  # if i have a week that starts in the last month ..
            # it will also end in the last month and hence fully be contained in it

            if (j == len(months_start_in_hours) - 1):
                if weeks_start_in_hours[i] >= months_start_in_hours[len(months_start_in_hours) - 1]:
                    continue  # if i am in the last month already ... this means the week is fully contained within
                    # the last month since there is no next month to jump to
                else:
                    exceptions.append(
                        i)  # if i reached the last month without finding a month that contains the week this means
                    # the week jumps from one month to the other
                    break
            elif (j < len(months_start_in_hours) - 1):
                if (months_start_in_hours[j] <= weeks_start_in_hours[i] < months_start_in_hours[j + 1]) and \
                        (months_start_in_hours[j] <= weeks_end_in_hours[i] < months_start_in_hours[j + 1]):
                    break


    #for i in exceptions:
    #    for j in range(len(months_start_in_hours) - 1):
    #        if (months_start_in_hours[j] <= weeks_start_in_hours[i] < months_start_in_hours[j + 1]):
    #            deledele = 0
    #            #print("week", str(i))
    #            #print(str(months_start_in_hours[j]), "<=", str(weeks_start_in_hours[i]), "<",
    #            #      str(months_start_in_hours[j + 1]))
    #            #print("But", str(weeks_end_in_hours[i]), ">=", str(months_start_in_hours[j + 1]))
    """
    from Building.py [[Jan Feb March Apr] , [May June July Aug Sept], [Oct Nov Dec] are the groups
    so [0,1,2,3] [4,5,6,7,8] [9,10,11]
    """

    # need to check if the exception starts at the end of a group ... otherwise no problem since
    # the only way to jump from a group to the other is by starting at the end of a group

    problem = []
    for i in exceptions:
        for j in (3, 8):
            if (months_start_in_hours[j] <= weeks_start_in_hours[i] < months_start_in_hours[j + 1]):
                #print(
                #    "Week number " + str(i) + " Starts in Month number " + str(j) + " and ends in Month number " + str(
                #        j + 1))
                #print(str(weeks_start_in_hours[i]), ">=", str(months_start_in_hours[j]))
                #print(str(weeks_end_in_hours[i]), ">=", str(months_start_in_hours[j + 1]))
                problem.append(i)

    test_results = {}
    for i in range(clustered["z"].shape[0]):
        list = []
        for j in range(clustered["z"].shape[1]):
            if clustered["z"][i][j] == 1:
                list.append((i, j))
            else:
                continue
        test_results[i] = list

    size_and_elements_of_cluster = {}
    T_e_mon_special = {}
    associate_weight = -1
    for i in range(clustered["z"].shape[0]):
        if len(test_results[i]) != 0:
            elements_of_the_cluster = []
            for j in range(len(test_results[i])):
                elements_of_the_cluster.append(test_results[i][j][1])
            associate_weight += 1
            x = clustered["weights"][associate_weight]
            T_e_mon_special[i] = T_e_mon[associate_weight]
            size_and_elements_of_cluster[i] = (x, elements_of_the_cluster)


    # for i in (size_and_elements_of_cluster.keys()):
    #     print(weeks_start_in_hours[i])
    #     for j in range(len(months_start_in_hours) - 1):
    #         if months_start_in_hours[j] <= weeks_start_in_hours[i] <= months_start_in_hours[j + 1]:
    #             print("week #", str(i), "in month #", str(j), "an has a weight of",
    #                   str(size_and_elements_of_cluster[i][0]))

    T_i_appr_special = {}
    for i in (size_and_elements_of_cluster.keys()):
        if i in problem:
            T_i_appr_special[i] = 27  # the problematic element is element 17 which
            # Week element number 17 Starts in Month element number 3 and ends in Month number element 4 .. (indexing starts at 0)
            # So in this case the week starts at hour 2857 the 30.04 and ends in may ... so only has one day of april in it
        else:
            for j in range(len(months_start_in_hours)):
                if j >= ((len(months_start_in_hours)) - 1):  # reached december
                    T_i_appr_special[i] = 20

                elif j < ((len(months_start_in_hours)) - 1) and months_start_in_hours[j] <= weeks_start_in_hours[i] < \
                        months_start_in_hours[j + 1]:
                    T_i_appr_special[i] = T_i_appr[j]
                    break

    list_size_and_elements_of_cluster = []
    for i in range(52):
        if i in size_and_elements_of_cluster.keys():
            x = size_and_elements_of_cluster[i]
            temp = (i, (x[0], x[1]))
            list_size_and_elements_of_cluster.append(temp)

    clustered["list_size_and_elements_of_cluster"] = list_size_and_elements_of_cluster

    # Heating period from October until May (important for T_i_appr)
    for i in range(number_of_representative_periods):
        j = list_size_and_elements_of_cluster[i][0]
        b_floor.append((T_i_year - T_e_year) / (T_i_appr_special[j] - T_e_mon_special[j]))
    b_floor = np.array(b_floor)

    b_tr = {}
    b_tr["wall"] = np.ones((number_of_representative_periods, length_of_cluster))
    b_tr["roof"] = np.ones((number_of_representative_periods, length_of_cluster))
    b_tr["floor"] = np.zeros((number_of_representative_periods, length_of_cluster))

    for i in range(number_of_representative_periods):
        b_tr["floor"][i, :] = b_floor[i]

    # upper and lower bound for T_s for linearization
    T_s_o = 50  # [°C]
    T_s_u = 0  # [°C]

    # upper and lower bound for T_m for linearization
    T_m_o = 50  # [°C]
    T_m_u = 0  # [°C]

    # initial temperature for building envelope
    T_init = 20  # [°C]

    # shadow coefficient for sunblinds
    #       (DIN EN ISO 13790, section 11.4.3, page 71)
    F_sh_gl = 1  # Assumption : no sunblinds (modelled manually, see below)

    # Dictionary for Irradiation to imitate sunblinds manually [kW/m²]
    I_sol = {}
    directions = ("south", "west", "north", "east", "roof")
    for drct3 in range(len(directions)):
        I_sol[directions[drct3]] = SunRad_part[drct3, :, :]
        I_sol["window", directions[drct3]] = SunRad_part[drct3, :, :]

    I_sol["floor"] = np.zeros_like(I_sol["roof"])
    I_sol["window", "floor"] = np.zeros_like(I_sol["roof"])

    limit_shut_blinds = 0.1  # kW/m²
    need_to_shut = []
    for i in (list_size_and_elements_of_cluster):
        x = i[0]
        if T_i_appr_special[x] == 27:  # this means  that the week belongs to the period may - sept
            need_to_shut.append(list_size_and_elements_of_cluster.index(i))  # now we have the position of that week

    for d in need_to_shut:  # May until September
        for t in range(length_of_cluster):
            for drct3 in range(len(directions)):  # For all directions
                if SunRad_part[drct3, d, t] > limit_shut_blinds:
                    I_sol["window", directions[drct3]][d, t] = 0.15 * SunRad_part[drct3, d, t]

    # 5R1C ISO 13790 End
    ###################

    # Economics
    ###########

    t_clc = 10.0  # Observation time period in years
    rate = 0.055  # Interest rate for annuity factor
    tax = 1.19  # Value added tax
    prChange = {"el": 1.057,  # Price change factors per year for electricity
                "gas": 1.041,  # Price change factors per year for natural gas
                "eex": 1.0252,  # Price change factors per year for eex compensation
                "infl": 1.017}  # Price change factors per year for inflation

    # Irradiation on 35° tilted surface (for "STC"-efficiency) in kW/m2
    solar_rad = SunRad_part[5, :, :]

    q = 1 + rate  # Abbreviation
    crf = (q ** t_clc * rate) / (q ** t_clc - 1)  # Capital recovery factor

    b = {key: (1 - (prChange[key] / q) ** t_clc) / (q - prChange[key])
         for key in prChange.keys()}  # Calculation of cash value B concerning energy price increase

    price = {"CHP": 0.04,  # Subsidy for electricity produced by CHP units from German co-generation act in EUR/kWh ...
             # from https://www.gesetze-im-internet.de/kwkg_2016/KWKG.pdf
             "el": 0.2660,  # Electricity price in EUR/kWh
             ("CHP", "sell"): 0.08,
             # Remuneration for sold electricity from CHP in EUR/kWh
             ("PV", "sell"): 0.1427,  # Remuneration for sold electricity from PV, fixed for 20 years
             ("gas", "CHP"): 0.0608,  # Natural gas price for CHP units without energy tax based on LHV in EUR/kWh
             ("gas", "Boiler"): 0.0693}  # Natural gas price for other applications based on LHV in EUR/kWh

    c_meter = {"gas": 157}  # Fixed costs for gas metering system EUR/year

    # Emission coefficients in kg_CO2 / kWh
    emi = {"PV": 0,
           "gas": 0.25,
           "el": 0.569}

    emission_max = max_emissions  # tons of CO2 per year

    # House related data:
    heat = {}  # Heat generation / consumption in kW (positive, device dependent)

    # Heat demand and domestic hot water demand in kW
    heat_dhw_raw = clustered["representative_days"]["dhw"] / 1000

    # Monthly averaging
    heat["DHW"] = np.zeros_like(T_e)
    for d in range(number_of_representative_periods):
        for t in range(length_of_cluster):
            heat["DHW"][d, t] = heat_dhw_raw[d, t]

    t_amb = T_e  # Ambient temperature for this region in °C

    # Common dictionaries for devices
    bounds = {}
    c_fix = {}
    eta = {}
    part_load = {}
    cap = {}
    f_serv = {}
    self_discharge = {}

    # Parameters for each device

    # Battery

    inv["Battery"] = 592.1  # Specific investment costs Battery in EUR/kWh
    c_fix["Battery"] = 5137  # Fix cost share Battery in EUR
    t_life["Battery"] = 13.7  # Based on number of cycles]
    bounds["Battery", "up"] = [3.3, 30]  # dummy in kWh # 5000 if testing seasonal storage
    bounds["Battery", "charge"] = 1 # 0.6286 for 1.6 hours  # Specific charging and discharging rates in kW/kWh ... 1 so that we can charge and discharge fully within 1 hour
    bounds["Battery", "discharge"] = 1 # 0.6286 for 1.6 hours  # Specific charging and discharging rates in kW/kWh ... 1 so that we can charge and discharge fully within 1 hour
    bounds["Battery", "DOD"] = 0.8
    eta["Battery", "charge"] = 0.8875  # eta_oneWay = sqrt(eta_cycle)
    eta["Battery", "discharge"] = 0.8875  # eta_oneWay = sqrt(eta_cycle)
    sub_battery = {("PV", "sell"): 0.6,  # Sold PV electricity must be <= 0.6 * ratedPower
                   ("PV", "sub"): 0.3,  # Maximum subsidy of 30 % of the investments
                   ("PV", "investment"): 1600.0,  # Refund 1600 EUR / kW_peak PV
                   ("PV", "bound"): 2000.0}  # Maximum refund 2000 EUR / kW_peak PV
    self_discharge["Battery"] = (6/100) / (30 * 24)  # (6/100) / (30 * 24)  #  6 Percent Losses per month converted to per Hour
    init = {}  # The initial state of charge is defined at a latter point for the  battery

    # Thermal energy storage
    # After continuous TES
    # The "i=0" term is for the already existing device
    # The "i=1" term is for the continuous devices
    inv["TES"] = [0, 726.9]  # EUR and EUR/m3
    cap["TES"] = [0.1, 2.0]  # capacity and maximum capacity in m³
    c_fix["TES"] = 447  # Fix cost share TES in EUR
    t_life["TES"] = 20
    bounds["TES", "up"] = max(cap["TES"])  # Upper bound for storage capacity in m3
    # bounds["TES", "lo"] = min(cap["TES"])  # Lower bound for storage capacity in m3
    bounds["TES", "lo"] = 0  # made it 0 because it was messing up my new runs because i want to
    # look at TES from 0 to max and because of this ... a restriction later is causing contradiction because it forces
    # The cap to be greater than 0
    deltaT = {"TES": 35,  # Maximum difference between upper and lower part of the storage unit in K
              "HP": 10}  # Reduced difference for heat pumps
    cp_water = 4180.0 / 3600000  # Specific heat capacity of water kWh/kgK
    rho_water = 975  # Density of water in kg/m3 at 75°C
    sto_loss = 0.006  # Storage losses 0.6 % per hour
    init["TES"] = 0  # Initialize storage content with 0 kWh

    # Boiler
    # The "i=0" term is for the already existing device
    # The "i=1" or more term is for the continuous devices

    inv["Boiler"] = [0, 51.56]  # Investment costs for boiler in EUR and EUR/m3
    cap["Boiler"] = [18, 37.5]  # Capacity and maximum capacity in kW
    c_fix["Boiler"] = 551.5  # Fix cost share Boiler in EUR
    part_load["Boiler"] = [0.3, 0.2]  # Minimal part load of boiler
    eta["Boiler"] = [0.7, 0.948]  # Thermal efficiency of a modern condensing boiler based on LHV
    f_serv["Boiler"] = 0.03  # Service costs based on total investment (c_fix + inv)
    t_life["Boiler"] = 18
    bounds["Boiler", "up"] = max(cap["Boiler"])  # Upper bound for boiler capacity in kW
    bounds["Boiler", "lo"] = 11  # Lower bound for boiler capacity in kW

    # CHP units

    inv["CHP"] = 2094  # Specific investment costs in Euro/kW for chp in Euro per kW electrical
    c_fix["CHP"] = 11291  # Fix cost share CHP in EUR
    cap["CHP"] = [5, 10]  # Not capacitites but rather 2 Upper bound for CHP capacity in kW to split it into  two
    # categories ... less than 5 and more than 5 KW ... so 2 decision variables (because of if cap<5 statement later
    # in the code)
    f_serv["CHP"] = 0.05  # Service costs based on investment
    t_life["CHP"] = 15
    bounds["CHP", "up"] = 10  # Upper bound for CHP capacity in kW
    eta["CHP", "total"] = 1.0  # Total efficiency of CHP unit using condensing technology
    eta["CHP", "el"] = 0.25  # Electrical efficiency CHP unit
    part_load["CHP"] = 0.5  # Minimal part load of CHP's with cap > 5000 W
    sigma = eta["CHP", "el"] / (eta["CHP", "total"] - eta["CHP", "el"])  # Power to heat ratio of CHP unit

    # Model in part load: P_el = alpha * Q_th_max + beta * Q_th + gamma [4, page 246, equation 4]
    params = {("CHP", "alpha"): -0.146,  # without unit
              ("CHP", "beta"): 0.66,  # without unit
              ("CHP", "gamma"): -2.620}  # kW

    # Photovoltaic system
    inv["PV"] = 1615.0  # Investment costs of PV system in EUR/kWp
    f_serv["PV"] = 65.0  # Service costs for the PV system in EUR/(kWp*year)
    t_life["PV"] = 20
    bounds["PV", "up"] = 50  # Maximum area on the roof covered with PV-cells in m2
    bounds["PV", "low"] = 8  # entspricht 1kWp
    eta["PV"] = 0.14 * 0.85  # Efficiency of the photovoltaic cells (0.14) and further losses (0.85 -> converter, wiring...)
    pv_specific_capacity = 0.125  # Specific peak power per square meter in kWp/m2

    # Heat Pump
    # Investment costs for air-water heat pumps computed by linear regression fit(MA Rheinhardt)

    inv["HP"] = 604.9  # Specific investment costs HP in EUR/kW
    f_serv["HP"] = 0.025  # Service costs based on investment
    t_life["HP"] = 18
    bounds["HP", "up"] = 25  # in kW
    c_fix["HP"] = 5316  # Fix cost share HP in EUR
    part_load["HP"] = 0.4  # Part load factor of HP
    t_flow = 35  # Set t_flow ONLY 35, 45 and 55 °C are valid!!
    t_return = 25

    # COP of HP at reference point A2W35
    if t_flow == 35:
        cop_ref = 3.8
    elif t_flow == 45:
        cop_ref = 2.96
    elif t_flow == 55:
        cop_ref = 2.44

    # Heat pump COP calculation:
    a_cop = 3.02367  # Parameter for COP curve
    b_cop = 126.96333  # Parameter for COP curve
    cop_lowest = (a_cop * T_ne + b_cop) / t_flow  # Calculation of COP at t_lowest
    cop = (a_cop * t_amb + b_cop) / t_flow  # Coefficient of performance of HP dependent on ambient temperature and t_flow

    # Electrical Heater
    inv["EH"] = 19  # Specific investment costs EH in EUR/kW_el
    c_fix["EH"] = 245  # Fix cost share electrical heater in EUR
    f_serv["EH"] = 0.0
    t_life["EH"] = 18
    bounds["EH", "up"] = 100  # dummy in kW
    eta["EH"] = 1  # Efficiency of electrical heater

    # Solar thermal collector
    inv["STC"] = 92.44  # EUR per m^2
    c_fix["STC"] = 596.24  # EUR fixed costs
    c_insurance = {"STC": 33.61}  # EUR per year (fixed costs)
    f_serv["STC"] = 0.015  # Service costs for solar thermal collectors. Based on initial investment
    t_life["STC"] = 30
    eta["STC", "optical"] = 0.8  # Optical efficiency, without unit
    eta["STC", "a1"] = 3.6390 / 1000  # kW per (m^2 K)
    eta["STC", "a2"] = 0.0168 / 1000  # kW per (m^2 K^2)

    t_stc_mean = (t_flow + t_return) / 2  # Mean temperature inside the collector

    eta["STC"] = (eta["STC", "optical"]
                  - eta["STC", "a1"] * (t_stc_mean - t_amb) / solar_rad
                  - eta["STC", "a2"] * ((t_stc_mean - t_amb) ** 2) / solar_rad)
    eta["STC"][np.isnan(eta["STC"])] = 0
    eta["STC"] = np.maximum(eta["STC"], 0)

    e_STC = np.zeros_like(eta["STC"])
    e_STC[eta["STC"] > 0] = 4.82 / 1000  # kW/m2 pumping electricity

    rval = {dev: (t_life[dev] - np.array([t_clc])) / t_life[dev] / (q ** t_clc)
            for dev in t_life.keys()}  # Calculation of residual value [3]

    # Additional parameters
    number_devices = {dev: len(cap[dev]) for dev in cap.keys()}  # Battery, CHP, HP, TES, Boiler

    ###############################################################################
    ###############################################################################
    # setting up the model
    # try:
    # Create a new model
    model = gp.Model("Energy system design for a single building")

    # Create new variables
    # Costs
    c_total = model.addVar(vtype="C", lb=-np.inf, ub=np.inf)
    c_inv = {dev: model.addVar(vtype="C")
             for dev in t_life.keys()}
    c_serv = {dev: model.addVar(vtype="C", name="c_serv_" + dev)
              for dev in f_serv.keys()}
    c_dem = {dev: model.addVar(vtype="C", name="c_dem_" + dev)
             for dev in ["Boiler", "HP", "CHP", "Grid"]}
    c_met = model.addVar(vtype="C", name="c_metering")

    # Devices

    cap_design = {dev: model.addVar(vtype="C", name="cap_" + dev)
                  for dev in ["EH", "Boiler", "Boiler_Conti", "TES", "TES_Conti", "PV",
                              "Battery", "Battery_small", "Battery_large", "HP", "CHP", "CHP0", "CHP1"]}
    initial = {dev: model.addVar(vtype="C", name="initial_" + dev) for dev in ["Battery", "Battery_small","Battery_large"]}

    meter = {dev: model.addVar(vtype="C", name="meter_" + dev)
             for dev in ("gas",)}

    revenue = {key: model.addVar(vtype="C", name="revenue_" + key)
               for key in ("CHP", "feed-in")}

    subsidy = model.addVar(vtype="C", name="subsidy")  # Battery purchase
    # If battery is installed, the exported PV power has to be limited (60%)
    limit_pv_battery = model.addVar(vtype="C", name="limit_pv_battery")

    emission = model.addVar(vtype="C", name="CO2_emissions", lb=-gp.GRB.INFINITY)

    emissions_gas1 = model.addVar(vtype="C", name="CO2_emissions_gas", lb=-gp.GRB.INFINITY)
    emissions_PV1 = model.addVar(vtype="C", name="CO2_emissions_PV", lb=-gp.GRB.INFINITY)
    emissions_grid1 = model.addVar(vtype="C", name="CO2_emissions_Grid", lb=-gp.GRB.INFINITY)
    emissions_lca1 = model.addVar(vtype="C", name="CO2_emissions_Grid_LCA", lb=-gp.GRB.INFINITY)

    energy = {}  # Energy consumption in Watt
    storage = {}  # Stored energy in Joule

    # ### Huge Change 2 to code old Start ###
    for t in range(length_of_cluster):
        for d in range(number_of_representative_periods):
            timetag = "_" + str(d) + "_" + str(t)
            for dev in ("TES", "Battery", "Battery_small", "Battery_large"):
                storage[dev, d, t] = model.addVar(vtype="C",
                                                  name="storage_" + dev + timetag)
            for dev in ("CHP", "Boiler"):
                energy[dev, d, t] = model.addVar(vtype="C",
                                                 name="energy_" + dev + timetag)

            for dev in ("Boiler", "CHP", "HP", "EH", "STC"):
                heat[dev, d, t] = model.addVar(vtype="C",
                                               name="heat_" + dev + timetag)

            for dev in ("PV", "CHP", "HP", "EH", "Import", "STC"):
                power[dev, d, t] = model.addVar(vtype="C",
                                                name="power_" + dev + timetag)

            for dev in ("Battery","Battery_small", "Battery_large"):
                power[dev, d, t, "charge"] = model.addVar(vtype="C",
                                                          name="power_" + dev + timetag + "_charge")
                power[dev, d, t, "discharge"] = model.addVar(vtype="C",
                                                             name="power_" + dev + timetag + "_discharge")

            for dev in ("PV", "CHP"):
                power[dev, d, t, "use"] = model.addVar(vtype="C",
                                                       name="power_" + dev + timetag + "_use")
                power[dev, d, t, "sell"] = model.addVar(vtype="C",
                                                        name="power_" + dev + timetag + "_sell")

    # These variables are mainly used for linearizations
    q_max = {}  # Heat related
    p_max = {}  # Power related
    s_max = {}  # Storage related
    for t in range(length_of_cluster):
        for d in range(number_of_representative_periods):
            timetag = "_" + str(d) + "_" + str(t)
            for dev in ("TES",):
                s_max[dev, d, t] = model.addVar(vtype="C", name="s_max_" + dev + timetag)

            for dev in ("HP",):
                q_max[dev, d, t] = model.addVar(vtype="C", name="q_max_" + dev + timetag)


            for dev in ("Boiler", "CHP"):
                for i in range(number_devices[dev]):
                    p_max[dev, d, t, i] = model.addVar(vtype="C",
                                                       name="p_max_" + dev + timetag + "_" + str(i))
                    q_max[dev, d, t, i] = model.addVar(vtype="C",
                                                       name="q_max_" + dev + timetag + "_" + str(i))

    x = {}  # Binary. 1 if device-type is installed, 0 otherwise
    for dev in number_devices.keys():
        for i in range(number_devices[dev]):
            x[dev, i] = model.addVar(vtype="B", name="x_" + dev + "_" + str(i))

    # My Addition


    for dev in ["PV", "STC", "EH", "Battery", "Battery_small", "Battery_large", "HP"]:
        x[dev] = model.addVar(vtype="B", name="x_" + dev)

    ###
    ###
    y = {}  # Binary. 1 if device is activated, 0 otherwise

    for d in range(number_of_representative_periods):
        for t in range(length_of_cluster):
            for dev in ("Boiler", "CHP"):
                for i in range(number_devices[dev]):
                    tag = dev + "_" + str(d) + "_" + str(t) + "_" + str(i)
                    y[dev, d, t, i] = model.addVar(vtype="B", name="y_" + tag)
            for dev in ("HP",):
                tag = dev + "_" + str(d) + "_" + str(t)
                y[dev, d, t] = model.addVar(vtype="B", name="y_" + tag)

    area = {dev: model.addVar(vtype="C", name="a_" + dev)
            for dev in ("PV", "STC")}

    # 5R1C ISO 13790
    ###############
    # 2.2) Create new variables
    # 2.2.1) decision variables x for each device of building envelope (Binary)

    for i in range(len(U["window"])):
        x["window", i] = model.addVar(vtype="B", name="x_window_" + str(i))
    for drct2 in direction2:
        for j in range(len(U["opaque", drct2])):
            x["opaque", drct2, j] = model.addVar(vtype="B",
                                                 name="x_opaque_" + drct2 + "_" + str(j))

    # binary variables for selection of building class (documentation z_cl)

    for l in range(len(f_class["Am"])):
        x["class", l] = model.addVar(vtype="B", name="x_class_" + str(l))

    # 2.2.2) Variables for linearization

    z_1 = {}
    z_2 = {}  # z_2[d,t,drct2,j] = x["opaque",drtc2,j] * T_m
    z_3 = {}  # z_3[d,t,i] = x["window",i] * T_s
    z_4 = {}
    z_5 = {}
    z_6 = {}  # z_6[d,t,l] = x["class",l] * T_m
    z_7 = {}  # z_7[d,t,l] = x["class",l] * T_s

    for d in range(number_of_representative_periods):
        for t in range(length_of_cluster):
            timetag = str(d) + "_" + str(t) + "_"
            for drct2 in direction2:
                for j in range(len(U["opaque", drct2])):
                    # 2.2.2.1) linearization of z_2[d,t, drct2, j] = x["opaque", drtc2, j] * T_m
                    z_2[d, t, drct2, j] = model.addVar(vtype="C",
                                                       name="z_2_" + timetag + drct2 + "_" + str(j))
            for i in range(len(U["window"])):
                # 2.2.2.2) linearization of z_3[d,t, i] = x["window", i] * T_s
                z_3[d, t, i] = model.addVar(vtype="C", name="z_3_" + timetag + str(i))
            for l in range(len(f_class["Am"])):
                # 2.2.2.3) linearization of z_6[d,t, l] = x["class", l] * T_m
                z_6[d, t, l] = model.addVar(vtype="C", name="z_6_" + timetag + str(l))
                # 2.2.2.4) linearization of z_7[d,t, l] = x["class", l] * T_s
                z_7[d, t, l] = model.addVar(vtype="C", name="z_7_" + timetag + str(l))

    for drct2 in direction2:
        for j in range(len(U["opaque", drct2])):
            for i in range(len(U["window"])):
                # 2.2.2.5) linearization of z_1[drct2, j, i] = x["window", i] * x["opaque", drct2, j]
                z_1[drct2, j, i] = model.addVar(vtype="B",
                                                name="z_1_" + drct2 + "_" + str(j) + "_" + str(i))
            for l in range(len(f_class["Am"])):
                # 2.2.2.6) linearization of z_4[drct2, j, l] = x["class", l] * x["opaque", drtc2, j]
                z_4[drct2, j, l] = model.addVar(vtype="B",
                                                name="z_4_" + drct2 + "_" + str(j) + "_" + str(l))

    for i in range(len(U["window"])):
        for l in range(len(f_class["Am"])):
            # 2.2.2.7) linearization of  z_5[i, l] = x["class", l] * x["window", i]
            z_5[i, l] = model.addVar(vtype="B", name="z_5_" + str(i) + "_" + str(l))

    # Continuous Variables for each timestep
    T_s = {}
    T_i = {}
    T_m = {}
    # T_m_init ={}
    Q_HC = {}  # in kW

    for d in range(number_of_representative_periods):
        for t in range(length_of_cluster):
            timetag = str(d) + "_" + str(t)
            # Surface related temperature (DIN EN ISO 13790, section 7.2.2.1, page 34 text)
            T_s[d, t] = model.addVar(vtype="C", name="T_s_" + timetag)
            # Room temperature
            T_i[d, t] = model.addVar(vtype="C", name="T_i_" + timetag)
            # Mass related temperature
            T_m[d, t] = model.addVar(vtype="C", name="T_m_" + timetag)
            # Heat demand
            Q_HC[d, t] = model.addVar(vtype="C", name="Q_HC_" + timetag,
                                      lb=0, ub=100)  # max. 100 kW

    # 5R1C ISO 13790 End
    ###################

    # Update model to integrate the new variables
    model.update()

    ###############################################################################
    ###############################################################################
    # Objective: Minimize investments, service, demand, metering costs (less
    #   generated revenues)
    if options["opt_costs"]:
        model.setObjective(c_total, gp.GRB.MINIMIZE)
    else:
        model.setObjective(emission, gp.GRB.MINIMIZE)

    ###############################################################################
    ###############################################################################
    # Constraints

    # Costs
    model.addConstr(c_total == sum(c_inv[dev] for dev in c_inv.keys()) +
                    sum(c_serv[dev] for dev in c_serv.keys()) +
                    sum(c_dem[dev] for dev in c_dem.keys()) +
                    c_met - sum(revenue[key] for key in revenue.keys()))
    # Investment costs:
    # ------------------------------------------------------------------------------

    for dev in ("TES", "Boiler"):
        model.addConstr(c_inv[dev] == crf * (1 - rval[dev][0]) * tax * (x[dev, 0] * inv[dev][0]
                                                                        + cap_design[dev + "_Conti"] * inv[dev][1]
                                                                        + x[dev, 1] * c_fix[dev]))
    for dev in ("CHP",):  # CHP is 2 distinct continuous variables now
        model.addConstr(c_inv[dev] == crf * (1 - rval[dev][0]) * tax * ((cap_design[dev + "0"] * inv[dev]
                                                                         + x[dev, 0] * c_fix[dev])
                                                                        + (cap_design[dev + "1"] * inv[dev]
                                                                           + x[dev, 1] * c_fix[dev])))
    for dev in ("HP",):
        model.addConstr(c_inv[dev] == crf * (1 - rval[dev][0]) * tax * (inv[dev] * cap_design[dev]
                                                                        + c_fix[dev] * x[dev]))

    for dev in ("EH",):
        model.addConstr(
            c_inv[dev] == crf * (1 - rval[dev][0]) * tax * (inv[dev] * cap_design[dev] + c_fix[dev] * x[dev]))

    dev = "PV"
    model.addConstr(c_inv[dev] == crf * (1 - rval[dev][0]) * tax * inv[dev] * cap_design[dev])
    dev = "STC"
    model.addConstr(c_inv[dev] == crf * (1 - rval[dev][0]) * tax * (inv[dev] * area[dev] + c_fix[dev] * x[dev]))
    dev = "Battery"

    model.addConstr(c_inv[dev] == crf * (1 - rval[dev][0]) * tax * (inv[dev] * cap_design[dev]
                                                                    + c_fix[dev] * x[dev]) - 1 / t_clc * subsidy)

    # 5R1C ISO 13790
    ###############

    for dev in direction2:
        model.addConstr(c_inv["opaque", dev] == crf * tax *
                        sum((1 - rval["opaque", dev][i]) * x["opaque", dev, i] * A["opaque", dev] * inv["opaque", dev][
                            i]
                            for i in range(len(inv["opaque", dev]))))
    dev = "window"
    model.addConstr(c_inv[dev] == crf * tax *
                    sum((1 - rval[dev][i]) * x[dev, i] * A[dev] * inv[dev][i]
                        for i in range(len(inv[dev]))))

    for dev in ("intWall", "ceiling", "intFloor"):
        model.addConstr(c_inv["opaque", dev] == 0.5 * crf * tax *
                        sum((1 - rval["opaque", dev][i]) * x["opaque", "wall", i] * A["opaque", dev] *
                            inv["opaque", dev][i]
                            for i in range(len(inv["opaque", dev]))))
    # 5R1C ISO 13790 End
    ###################
    # Invest battery
    # Battery subsidies
    dev = "Battery"

    model.addConstr(subsidy <= sub_battery["PV", "sub"] *
                    ((inv[dev] * cap_design[dev] + c_fix[dev] * x[dev]) +
                     (inv["PV"] - sub_battery["PV", "investment"]) * cap_design["PV"]))
    model.addConstr(subsidy <= sub_battery["PV", "sub"] * sub_battery["PV", "bound"] * cap_design["PV"])
    model.addConstr(subsidy <= sub_battery["PV", "sub"] * sub_battery["PV", "bound"] * x[dev])
    #
    # Service and operation costs
    # ------------------------------------------------------------------------------
    dev = "PV"

    model.addConstr(c_serv[dev] == crf * b["infl"] * tax * f_serv[dev] * cap_design[dev])

    dev = "STC"
    model.addConstr(c_serv[dev] == crf * b["infl"] * tax *
                    (f_serv[dev] * (inv[dev] * area[dev] + c_fix[dev] * x[dev]) + c_insurance["STC"] * x["STC"]))

    for dev in ("CHP",):
        model.addConstr(c_serv[dev] == crf * b["infl"] * tax * f_serv[dev] * ((cap_design[dev + "0"] * inv[dev]
                                                                               + x[dev, 0] * c_fix[dev])
                                                                              + (cap_design[dev + "1"] * inv[dev]
                                                                                 + x[dev, 1] * c_fix[dev])))

    for dev in ("HP",):
        model.addConstr(c_serv[dev] == crf * b["infl"] * tax * f_serv[dev]
                        * (cap_design[dev] * inv[dev] + x[dev] * c_fix[dev]))

    for dev in ("Boiler",):
        model.addConstr(c_serv[dev] == crf * b["infl"] * tax * f_serv[dev]
                        * (x[dev, 0] * inv[dev][0] + cap_design[dev + "_Conti"] * inv[dev][1] + x[dev, 1] * c_fix[dev]))


    # Demand related costs
    # ------------------------------------------------------------------------------

    for dev in ("Boiler", "CHP"):
        model.addConstr(c_dem[dev] == crf * b["gas"] * price["gas", dev] * dt *
                        sum(clustered["weights"][d] *
                            sum(energy[dev, d, t] for t in range(length_of_cluster))
                            for d in range(number_of_representative_periods)))
    model.addConstr(c_dem["Grid"] == crf * b["el"] * price["el"] * dt *
                    sum(clustered["weights"][d] *
                        sum(power["Import", d, t] for t in range(length_of_cluster))
                        for d in range(number_of_representative_periods)))

    # Costs for metering
    # ------------------------------------------------------------------------------
    model.addConstr(c_met == crf * b["infl"] * (c_meter["gas"] * meter["gas"]))

    # Revenue
    # ------------------------------------------------------------------------------

    model.addConstr(revenue["CHP"] == crf * b["eex"] * price["CHP"] * dt *
                    sum(clustered["weights"][d] *
                        sum(power["CHP", d, t] for t in range(length_of_cluster))
                        for d in range(number_of_representative_periods)))
    model.addConstr(revenue["feed-in"] == crf * b["eex"] * dt *
                    sum(clustered["weights"][d] *
                        sum(power["CHP", d, t, "sell"] * price["CHP","sell"] +
                            power["PV", d, t, "sell"] * price["PV", "sell"]
                            for t in range(length_of_cluster))
                        for d in range(number_of_representative_periods)))

    # Metering constraints
    for dev in ("Boiler", "CHP"):
        model.addConstr(meter["gas"] >= sum(x[dev, i] for i in range(number_devices[dev])))

    # CO2 emissions

    emissions_gas = emi["gas"] * sum(clustered["weights"][d] * dt *
                                     sum(sum(energy[dev, d, t]
                                             for dev in ("Boiler", "CHP"))
                                         for t in range(length_of_cluster))
                                     for d in range(number_of_representative_periods))
    emissions_PV = emi["PV"] * sum(clustered["weights"][d] * dt *
                                   sum(power["PV", d, t] for t in range(length_of_cluster))
                                   for d in range(number_of_representative_periods))

    emissions_grid = 0
    for d in range(number_of_representative_periods):
        emi_cluster = 0
        for t in range(length_of_cluster):
            factor = clustered["representative_days"]["CO2(t)"][d][t]
            emi_cluster += (power["Import", d, t]) * factor
        emissions_grid += clustered["weights"][d] * emi_cluster

    # Emissions due to the components them selves (LCA)

    emi_lca_materials = {"ferroconcrete": 237.34,  # kg CO2 Äqu/ m3
                         "polystyrene": 55.27,  # kg CO2 Äqu/ m3
                         "concrete layer for outer walls": 242,  # kg CO2 Äqu/ m3
                         "concrete layer": 242,  # kg CO2 Äqu/ m3
                         "gypsum plasterboard": 1.5745,  # kg CO2 Äqu/ m2
                         "sandwich panel": 10.73,  # kg CO2 Äqu/ m3
                         "core insulation": 55.27,  # kg CO2 Äqu/ m3
                         "air gap": 0,  # kg CO2 Äqu/ m3
                         "lime stone": 303.4,  # kg CO2 Äqu/ m3
                         "Plastering": 718.4,  # kg CO2 Äqu/ m3
                         "Lime Plaster": 405.1,  # kg CO2 Äqu/ m3
                         "concrete for internal wall": 69.2,  # # kg CO2 Äqu/ m2
                         "concrete for ceiling": 242,  # kg CO2 Äqu/ m3
                         "cement screed": 367.2,  # kg CO2 Äqu/ m3
                         "concrete for ground floor": 237.34,  # kg CO2 Äqu/ m3
                         "foam glas": 158.4,  # kg CO2 Äqu/ m3
                         "gravel fill": 32.42,  # kg CO2 Äqu/ m3
                         "insulation board": 266,  # kg CO2 Äqu/ m3
                         "Wooden Frame": -0.26,  # kg CO2 Äqu/ m
                         "double glazing": 26.7,  # kg CO2 Äqu/ m2
                         "Plastic frame": 8.17,  # kg CO2 Äqu/ m
                         "insulation glazing": 9.87,  # kg CO2 Äqu/ m2
                         "double insulation glazing": 37,  # kg CO2 Äqu/ m2
                         "Triple insulation glazing": 50.16  # kg CO2 Äqu/ m2
                         }

    Area_outside_walls = 4 * 42.25  # m2
    Area_int_walls = 375  # m2
    Area_roof = 99.75  # m2
    Area_floor = 99.75  # m2
    A_ceiling = 75  # m2
    A_int_floor = 75  # m2
    Area_windows = 4 * 7.5  # m2

    ow_0 = 0.175 * emi_lca_materials["ferroconcrete"] + 0.03 * emi_lca_materials["polystyrene"] + 0.08 * \
           emi_lca_materials["concrete layer for outer walls"]
    ow_1 = ow_0 + 0.0125 * emi_lca_materials["gypsum plasterboard"] + 0.04 * emi_lca_materials["sandwich panel"]
    ow_2 = ow_0 + 0.008 * emi_lca_materials["Plastering"] + 0.15 * emi_lca_materials["core insulation"] + 0.01 * \
           emi_lca_materials["air gap"] + 0.115 * emi_lca_materials["lime stone"]
    ow_3 = ow_2 + 0.07 * emi_lca_materials["core insulation"]
    iw_0 = 0.01 * emi_lca_materials["Lime Plaster"] + 0.10 * emi_lca_materials["concrete for internal wall"] + 0.01 * \
           emi_lca_materials["Lime Plaster"]
    ceil_0 = 0.16 * emi_lca_materials["concrete for ceiling"] + 0.06 * emi_lca_materials["polystyrene"] + 0.04 * \
             emi_lca_materials["cement screed"]
    flr_0 = 0.04 * emi_lca_materials["cement screed"] + 0.06 * emi_lca_materials["polystyrene"] + 0.16 * \
            emi_lca_materials["concrete layer"]
    gf_0 = 0.04 * emi_lca_materials["cement screed"] + 0.03 * emi_lca_materials["polystyrene"] + 0.15 * \
           emi_lca_materials["concrete for ground floor"]
    rf_0 = 0.15 * emi_lca_materials["ferroconcrete"] + 0.07 * emi_lca_materials["foam glas"] + 0.03 * emi_lca_materials[
        "gravel fill"]
    rf_1 = 0.15 * emi_lca_materials["ferroconcrete"] + 0.09 * emi_lca_materials["foam glas"] + 0.03 * emi_lca_materials[
        "gravel fill"]
    rf_2 = 0.15 * emi_lca_materials["ferroconcrete"] + 0.24 * emi_lca_materials["insulation board"] + 0.03 * \
           emi_lca_materials["gravel fill"]
    rf_3 = 0.15 * emi_lca_materials["ferroconcrete"] + 0.36 * emi_lca_materials["insulation board"] + 0.03 * \
           emi_lca_materials["gravel fill"]
    win_0 = emi_lca_materials["Wooden Frame"] + Area_windows * emi_lca_materials[
        "double glazing"]  # Total window area = 30 m squared from Building.py line 20
    win_1 = emi_lca_materials["Plastic frame"] + Area_windows * emi_lca_materials["insulation glazing"]
    win_2 = emi_lca_materials["Plastic frame"] + Area_windows * emi_lca_materials["double insulation glazing"]
    win_3 = emi_lca_materials["Plastic frame"] + Area_windows * emi_lca_materials["Triple insulation glazing"]
    factor_STC = 104.3  # kg CO2 Äqu/ m2
    factor_EH = 0.71  # kg CO2 Äqu/ kw
    factor_PV = 304  # kg CO2 Äqu/ m2
    factor_Battery = 243.9  # 243.9 #0  # kg CO2 Äqu/ kwh
    factor_TES = 3.60 * 200 # kg CO2 Äqu/ m3
    # For TES assumed a cylinder with radius R and height H=2R made of Stainless steehl (7900 kg/m3) and a 5mm tchick wall
    # From Ökobaudat 3,6 kg CO2 Äqu per kg steel
    # this would give:
    # V     R      H        Mass Steel       Mass Steel after Assumption y=200x
    #0,1    0,25    0,5     46,5                20
    #0,3    0,36    0,72    94,49               60
    #0,5    0,43    0,86    137,66              100
    #1      0,54    1,08    217                 200
    #2      0,68    1,36    344                 359
    # Linear regression of Volume and Steehl mass gives y = 172 x + 15
    # not feasible to impliment constant factor and variable one together for LCA
    # so gona use y = 200 x instead (mass of steel in kg = 200 * volume TES in m3
    # This  is why factor TES is multiplied by 200 ... that way in the dictionnary below directly multipled by volume of TES
    factor_Boiler = 21.40  # kg CO2 Äqu/ kw
    factor_HP = 45.30  # kg CO2 Äqu/ kw
    factor_CHP = 21.40  # kg CO2 Äqu/ kwel

    emi_lca_components = {"Outer Wall 0": 0 * (Area_outside_walls * ow_0) / 40,
                          # Life time of walls = 40 years ... already installed so 0 emissions
                          "Outer Wall 1": (Area_outside_walls * ow_1) / 40,  # Life time of walls = 40 years
                          "Outer Wall 2": (Area_outside_walls * ow_2) / 40,  # Life time of walls = 40 years
                          "Outer Wall 3": (Area_outside_walls * ow_3) / 40,  # Life time of walls = 40 years
                          # "Internal Wall": iw_0, # always there ... not an option to be added or saniert
                          # "Ceiling": ceil_0, # always there ... not an option to be added or saniert
                          "Floor": 0 * (Area_floor * flr_0) / 40,
                          # Life time of Floor = 40 years ... already installed so 0 emissions
                          # "Ground Floor": # always there ... not an option to be added or saniert
                          "Roof 0": 0 * (Area_roof * rf_0) / 40,
                          # Life time of roof = 40 years  ... already installed so 0 emissions
                          "Roof 1": (Area_roof * rf_1) / 40,  # Life time of roof = 40 years
                          "Roof 2": (Area_roof * rf_2) / 40,  # Life time of roof = 40 years
                          "Roof 3": (Area_roof * rf_3) / 40,  # Life time of roof = 40 years
                          "Windows 0": 0 * (win_0) / 40,
                          # Life time of windows = 40 years ... already installed so 0 emissions
                          "Windows 1": (win_1) / 40,  # Life time of windows = 40 years
                          "Windows 2": (win_2) / 40,  # Life time of windows = 40 years
                          "Windows 3": (win_3) / 40,  # Life time of windows = 40 years
                          "Battery": (cap_design["Battery"] * factor_Battery) / 13.7,  #
                          "TES 0": 0,  # Already installed so 0 emissions
                          "TES 1": (cap_design["TES_Conti"] * factor_TES) / 20,  # Life time of TES = 20 years
                          "Boiler 0": 0,  # Already installed so 0 emissions
                          "Boiler 1": (cap_design["Boiler_Conti"] * factor_Boiler) / 18,  # Life time of
                          "CHP 0": (cap_design["CHP0"] * factor_CHP) / 15,  # Life time of CHP = 15 years
                          "CHP 1": (cap_design["CHP1"] * factor_CHP) / 15,  # Life time of CHP = 15 years
                          "PV": (area["PV"] * factor_PV) / 20,  # Life time of PV = 20
                          "HP": (cap_design["HP"] * factor_HP) / 18,  # Life time of HP = 18
                          "EH": (cap_design["EH"] * factor_EH) / 18,  # Life time of EH = 18
                          "STC": (area["STC"] * factor_STC) / 30,
                          }

    decision_variable = {"Outer Wall 0": x["opaque", "wall", 0],
                         "Outer Wall 1": x["opaque", "wall", 1],
                         "Outer Wall 2": x["opaque", "wall", 2],
                         "Outer Wall 3": x["opaque", "wall", 3],
                         # "Internal Wall": 1 , # always there ... not an option to be added or saniert
                         # "Ceiling": 1 , # always there ... not an option to be added or saniert
                         "Floor": x["opaque", "floor", 0],
                         # "Ground Floor": gf_0, # always there ... not an option to be added or saniert
                         "Roof 0": x["opaque", "roof", 0],
                         "Roof 1": x["opaque", "roof", 1],
                         "Roof 2": x["opaque", "roof", 2],
                         "Roof 3": x["opaque", "roof", 3],
                         "Windows 0": x["window", 0],
                         "Windows 1": x["window", 1],
                         "Windows 2": x["window", 2],
                         "Windows 3": x["window", 3],
                         "Battery": x["Battery"],
                         "TES 0": x["TES", 0],
                         "TES 1": x["TES", 1],
                         "Boiler 0": x["Boiler", 0],
                         "Boiler 1": x["Boiler", 1],
                         "CHP 0": x["CHP", 0],
                         "CHP 1": x["CHP", 1],
                         "PV": x["PV"],
                         "HP": x["HP"],
                         "EH": x["EH"],
                         "STC": x["STC"],
                         }
    emissions_lca = 0

    for component in emi_lca_components.keys():
        if component in ["EH", "Battery", "STC", "PV", "TES 1", "Boiler 1", "CHP 0", "CHP 1", "HP"]:
            emissions_lca += (emi_lca_components[
                component])  # No need to convert from kg to tonnes here because it is done at the end with the .X method/ 1000  # Convert from kgs to Tonnes
        else:
            emissions_lca += (decision_variable[component] * emi_lca_components[
                component])  # No need to convert from kg to tonnes here because it is done at the end with the .X method/ 1000  # Convert from kgs to Tonnes

    model.addConstr(emission == emissions_gas + emissions_PV + emissions_grid + emissions_lca)

    model.addConstr(emissions_gas1 == emissions_gas)
    model.addConstr(emissions_PV1 == emissions_PV)
    model.addConstr(emissions_grid1 == emissions_grid)
    model.addConstr(emissions_lca1 == emissions_lca)

    model.addConstr(0.001 * emission <= emission_max)

    # Technology related and logical constraints
    # ------------------------------------------------------------------------------
    # Operation and design of devices
    # Design of devices (constraints without time dependency)

    # Boiler
    # included with the calculation of cap_design TES in the lines below in the equations of thermal storage

    # Electrical heater
    model.addConstr(cap_design["EH"] <= bounds["EH", "up"] * x["EH"])


    # To remove illogical negative capacity
    # Not necessary anymore because turns out gurobi variables are by default positive unless explicitly
    # specified that a ceratin variable can take negative values
    model.addConstr(cap_design["EH"] >= 0)
    model.addConstr(cap_design["HP"] >= 0)
    model.addConstr(cap_design["CHP"] >= 0)
    model.addConstr(cap_design["Boiler" + "_Conti"] >= 0)
    model.addConstr(cap_design["TES" + "_Conti"] >= 0)
    model.addConstr(area["STC"] >= 0)
    model.addConstr(cap_design["Boiler" + "_Conti"] <= cap["Boiler"][1] * x["Boiler", 1])

    model.addConstr(cap_design["Battery"] == cap_design["Battery_small"] + cap_design["Battery_large"])
    model.addConstr(cap_design["Battery_small"] <= bounds["Battery", "up"][0] * x["Battery_small"])
    model.addConstr(cap_design["Battery_large"] <= bounds["Battery", "up"][1] * x["Battery_large"])
    model.addConstr(cap_design["Battery_small"] >= 0)
    model.addConstr(cap_design["Battery_large"] >= x["Battery_large"] * 1.001 * bounds["Battery", "up"][0])

    model.addConstr(cap_design["CHP0"] <= cap["CHP"][0] * x["CHP", 0])
    model.addConstr(cap_design["CHP1"] <= cap["CHP"][1] * x["CHP", 1])
    model.addConstr(cap_design["CHP" + "0"] >= 0)
    model.addConstr(cap_design["CHP" + "1"] >= ((cap["CHP"][0] + 0.001) * x["CHP", 1]))
    model.addConstr(cap_design["CHP"] == cap_design["CHP0"] + cap_design["CHP1"])

    model.addConstr(cap_design["HP"] <= bounds["HP", "up"] * x["HP"])

    for dev in ("TES", "Boiler"):
        model.addConstr(cap_design[dev + "_Conti"] <= bounds[dev, "up"] * x[dev, 1])
        model.addConstr(cap_design[dev + "_Conti"] >= bounds[dev, "lo"] * x[dev, 1])

    # Thermal storage

    for dev in ("TES", "Boiler"):
        model.addConstr(cap_design[dev] == x[dev, 0] * cap[dev][0] + cap_design[dev + "_Conti"])
        model.addConstr(sum(x[dev, i] for i in range(number_devices["TES"])) <= 1)

    # Battery
    dev = "Battery"

    sum_x_bat = x["Battery_small"] + x["Battery_large"]
    model.addConstr(sum_x_bat <= 1)
    model.addConstr(x["Battery"] == sum_x_bat)

    # Linearization of limit_pv_battery = x["Battery",i]*cap_design["PV"]
    model.addConstr(
        limit_pv_battery <= sum_x_bat * min(bounds["PV", "up"], A["opaque", "floor"]) * pv_specific_capacity)
    model.addConstr(cap_design["PV"] - limit_pv_battery >= 0)
    model.addConstr(cap_design["PV"] - limit_pv_battery <=
                    (1 - sum_x_bat) * min(bounds["PV", "up"], A["opaque", "floor"]) * pv_specific_capacity)

    # PV and STC
    for dev in area.keys():
        model.addConstr(area[dev] <= min(bounds["PV", "up"], A["opaque", "floor"]) * x[dev])
    model.addConstr(area["PV"] + area["STC"] <= min(bounds["PV", "up"], A["opaque", "floor"]))
    model.addConstr(area["PV"] * pv_specific_capacity == cap_design["PV"])
    model.addConstr(area["PV"] >= bounds["PV", "low"] * x["PV"])

    ###
    # 5R1C ISO 13790
    ###############

    # Nominal heat load
    # Q_nHC = Q_T,i + Q_v_i + Q_RH,i (REG1 VL04)
    # Q_RH,i = A_f * f_RH not considered!!!
    Q_nHC = (sum(A["opaque", "wall"] * U["opaque", "wall"][i] * x["opaque", "wall", i]
                 for i in range(len(U["opaque", "wall"]))) +
             sum(A["window"] * U["window"][i] * x["window", i]
                 for i in range(len(U["window"]))) +
             sum(A["opaque", "roof"] * U["opaque", "roof"][i] * x["opaque", "roof", i]
                 for i in range(len(U["opaque", "roof"]))) +
             sum(A["opaque", "floor"] * U["opaque", "floor"][i] * x["opaque", "floor", i]
                 for i in range(len(U["opaque", "floor"]))) * f_g1 * f_g2 * G_w
             + ventilationRate * c_p_air * rho_air * V / 3600) * (T_set_min - T_ne)

    model.addConstr(((cap_design["CHP"] / sigma)) +
                    cap_design["Boiler"] + cap_design["EH"] * eta["EH"] +
                    cop_lowest / cop_ref * cap_design["HP"] >= 0.001 * Q_nHC)

    # 5R1C ISO 13790 End
    ###################
    ###
    ###
    # HP, CHP

    for dev in ("HP",):
        # I think this statement is redundant since x is a gurobi binary variable by definition
        # So this means it is always either 0 or 1
        # But it does not hurt to add it ... it still is true
        model.addConstr(x[dev] <= 1)

    for dev in ("CHP",):
        model.addConstr(sum(x[dev, i] for i in range(number_devices[dev])) <= 1)

    q_CHP = [(cap_design["CHP" + str(i)] - (params["CHP", "gamma"] * x["CHP", i])) /
             (params["CHP", "alpha"] + params["CHP", "beta"])
             for i in range(len(cap["CHP"]))]
    # here i multiplied params gamma by x chp in order to make the numerator 0 if cap design is 0
    # because of the way this factor is used later in the calculation of q_maxCHP1
    # and the equation is still linear because params gamma is a constant

    #for i in range(len(cap["CHP"])):
    #    xx = q_CHP[i]
    #    deledele = 0
    #    #print(xx)

    for d in range(number_of_representative_periods):
        # Storage cycling
        model.addConstr(storage["TES", d, length_of_cluster - 1] == init["TES"])
        dev = "Battery"

        model.addConstr(initial["Battery_small"] == 0.5 * cap_design["Battery_small"])
        model.addConstr(storage["Battery_small", d, length_of_cluster - 1] == initial["Battery_small"])
        model.addConstr(initial["Battery_large"] == 0.5 * cap_design["Battery_large"])
        model.addConstr(storage["Battery_large", d, length_of_cluster - 1] == initial["Battery_large"])

        for t in range(length_of_cluster):
            # Electricity balance (generation = consumption)

            model.addConstr(power["Import", d, t] +
                            sum(power[generator, d, t, "use"] for generator in ("CHP", "PV")) +
                            power["Battery", d, t, "discharge"] ==
                            sum(power[consumer, d, t] for consumer in ("HP", "EH", "STC")) +
                            clustered["representative_days"]["electricity"][d, t] +
                            power["Battery", d, t, "charge"])

            # Thermal storage
            # Energy balance
            if t > 0:
                storage_previous = storage["TES", d, t - 1]
                battery_previous_small = storage["Battery_small", d, t - 1]
                battery_previous_large = storage["Battery_large", d, t - 1]
            else:
                storage_previous = init["TES"]
                dev = "Battery"

                battery_previous_small = initial["Battery_small"]
                battery_previous_large = initial["Battery_large"]

            model.addConstr(storage["TES", d, t] == (1 - sto_loss) * storage_previous +
                            dt * (sum(heat[dev, d, t] for dev in ("Boiler", "CHP", "HP", "EH", "STC"))) -
                            dt * (Q_HC[d, t]) - dt * heat["DHW"][d, t])

            # Design restriction
            model.addConstr(storage["TES", d, t] <= cap_design["TES"] * rho_water * cp_water * deltaT["TES"])
            # Battery storage
            dev = "Battery"

            model.addConstr(power[dev, d, t, "charge"] ==  (power["Battery_small", d, t, "charge"] + power["Battery_large", d, t, "charge"] ))

            model.addConstr(power[dev, d, t, "discharge"] == (power["Battery_small", d, t, "discharge"] + power["Battery_large", d, t, "discharge"]))

            model.addConstr(storage[dev, d, t] == storage["Battery_small", d, t] + storage["Battery_large", d, t])

            model.addConstr(storage["Battery_small", d, t] >= (1 - bounds["Battery", "DOD"]) * cap_design["Battery_small"])
            model.addConstr(storage["Battery_small", d, t] <= cap_design["Battery_small"])
            model.addConstr(storage["Battery_small", d, t] == battery_previous_small +
                            dt * ((eta["Battery", "charge"] * power["Battery_small", d, t, "charge"]) - ((1 / eta["Battery", "discharge"]) * power["Battery_small", d, t, "discharge"]))
                            - (self_discharge["Battery"] * battery_previous_small))

            model.addConstr(storage["Battery_large", d, t] >= (1 - bounds["Battery", "DOD"]) * cap_design["Battery_large"])
            model.addConstr(storage["Battery_large", d, t] <= cap_design["Battery_large"])

            model.addConstr(storage["Battery_large", d, t] == battery_previous_large +
                            dt * ((eta["Battery", "charge"] * power["Battery_large", d, t, "charge"]) - (
                        (1 / eta["Battery", "discharge"]) * power["Battery_large", d, t, "discharge"]))
                            - (self_discharge["Battery"] * battery_previous_large))

            model.addConstr(power["Battery_small", d, t, "charge"] <= bounds["Battery", "charge"] * cap_design["Battery_small"])
            model.addConstr(power["Battery_large", d, t, "charge"] <= bounds["Battery", "up"][0] * x["Battery_large"])

            model.addConstr(power["Battery_small", d, t, "discharge"] <= bounds["Battery", "discharge"] * cap_design["Battery_small"])
            model.addConstr(power["Battery_large", d, t, "discharge"] <= bounds["Battery", "up"][0] * x["Battery_large"])

            # Electrical heater
            # Heat output
            model.addConstr(heat["EH", d, t] == power["EH", d, t] * eta["EH"])
            # Design restriction
            model.addConstr(cap_design["EH"] >= power["EH", d, t])

            # PV unit
            # Power output
            model.addConstr(power["PV", d, t] <= area["PV"] * eta["PV"] * solar_rad[d, t])
            model.addConstr(power["PV", d, t] == power["PV", d, t, "use"] + power["PV", d, t, "sell"])
            model.addConstr(cap_design["PV"] >= power["PV", d, t])
            # Limit sold power if battery is installed:
            model.addConstr(power["PV", d, t, "sell"] <= cap_design["PV"] -
                            (1 - sub_battery["PV", "sell"]) * limit_pv_battery)

            # Solar thermal heater
            model.addConstr(heat["STC", d, t] <= area["STC"] * solar_rad[d, t] * eta["STC"][d, t])
            model.addConstr(power["STC", d, t] == area["STC"] * e_STC[d, t])

            # Heat pump
            # Heat output
            # Linearize: sum(y["HP",d,t,i] for i) * cap_design["TES"] = s_max["TES",d,t]

            model.addConstr(s_max["TES", d, t] <= y["HP", d, t] * bounds["TES", "up"])
            model.addConstr(cap_design["TES"] - s_max["TES", d, t] >= 0)
            model.addConstr(cap_design["TES"] - s_max["TES", d, t] <= (1 - y["HP", d, t]) * bounds["TES", "up"])

            model.addConstr(storage["TES", d, t] <= rho_water * cp_water * (deltaT["HP"] - deltaT["TES"])
                            * s_max["TES", d, t] + rho_water * cp_water * deltaT["TES"] * cap_design["TES"])

            model.addConstr(q_max["HP", d, t] == cop[d, t] / cop_ref * cap_design["HP"])
            model.addConstr(cap_design["HP"] <= bounds["HP", "up"] * y["HP", d, t])
            model.addConstr(heat["HP", d, t] <= q_max["HP", d, t])
            model.addConstr(heat["HP", d, t] >= q_max["HP", d, t] * part_load["HP"])
            model.addConstr(power["HP", d, t] == heat["HP", d, t] / cop[d, t])
            model.addConstr(y["HP", d, t] <= x["HP"])

            # CHP unit

            model.addConstr(heat["CHP", d, t] == sum(q_max["CHP", d, t, i] for i in range(number_devices["CHP"])))
            model.addConstr(power["CHP", d, t] == sum(p_max["CHP", d, t, i] for i in range(number_devices["CHP"])))
            model.addConstr(power["CHP", d, t] == power["CHP", d, t, "use"] + power["CHP", d, t, "sell"])
            model.addConstr(energy["CHP", d, t] == 1.0 / eta["CHP", "total"] * (heat["CHP", d, t] + power["CHP", d, t]))
            for i in range(number_devices["CHP"]):
                model.addConstr(y["CHP", d, t, i] <= x["CHP", i])
            # if cap["CHP"][i] < 5:  # According to Spieker and Tsatsaronis
            model.addConstr(q_max["CHP", d, t, 0] == cap_design["CHP0"] / sigma)
            model.addConstr(p_max["CHP", d, t, 0] == cap_design["CHP0"])

            # else:
            model.addConstr(p_max["CHP", d, t, 1] <= cap_design["CHP1"])
            model.addConstr(p_max["CHP", d, t, 1] >= cap_design["CHP1"] * part_load["CHP"])
            model.addConstr(q_max["CHP", d, t, 1] == 1.0 / params["CHP", "beta"]
                            * (p_max["CHP", d, t, 1] - (
                        params["CHP", "alpha"] * q_CHP[1] + params["CHP", "gamma"] * x["CHP", i])))

            # Boiler

            model.addConstr(
                heat["Boiler", d, t] == sum(q_max["Boiler", d, t, i] for i in range(number_devices["Boiler"])))
            model.addConstr(energy["Boiler", d, t] == sum(1.0 / eta["Boiler"][i] * q_max["Boiler", d, t, i]
                                                          for i in range(number_devices["Boiler"])))

            for i in range(number_devices["Boiler"]):
                model.addConstr(y["Boiler", d, t, i] <= x["Boiler", i])

            model.addConstr(q_max["Boiler", d, t, 0] <= y["Boiler", d, t, 0] * cap["Boiler"][0])
            model.addConstr(q_max["Boiler", d, t, 1] <= cap_design["Boiler_Conti"])
            model.addConstr(q_max["Boiler", d, t, 0] >= y["Boiler", d, t, 0] * cap["Boiler"][0]
                            * part_load["Boiler"][0])
            model.addConstr(q_max["Boiler", d, t, 1] >= cap_design["Boiler_Conti"]
                            * part_load["Boiler"][1])

    ###
    # 5R1C ISO 13790
    ###############
    # Constraints
    # Decision Variables: only one type activated
    model.addConstr(sum(x["window", i] for i in range(len(U["window"]))) == 1, "window")

    for drct2 in direction2:
        model.addConstr(sum(x["opaque", drct2, j] for j in range(len(U["opaque", drct2])))
                        == 1, "opaque_" + drct2)

    model.addConstr(sum(x["class", l] for l in range(len(f_class["Am"]))) == 1, "class")

    # 2.4.2) linearizations
    for d in range(number_of_representative_periods):
        for t in range(length_of_cluster):
            timetag = str(d) + "_" + str(t)
            for drct2 in direction2:
                for j in range(len(U["opaque", drct2])):
                    # z_2[t, drct2, j] = x["opaque", drtc2, j] * T_m
                    model.addConstr(z_2[d, t, drct2, j] <= x["opaque", drct2, j] * T_m_o,
                                    "lin1_z_2_" + timetag + "_" + drct2 + "_" + str(j))
                    model.addConstr(z_2[d, t, drct2, j] >= x["opaque", drct2, j] * T_m_u,
                                    "lin2_z_2_" + timetag + "_" + drct2 + "_" + str(j))
                    model.addConstr(T_m[d, t] - z_2[d, t, drct2, j] <= (1 - x["opaque", drct2, j]) * T_m_o,
                                    "lin3_z_2_" + timetag + "_" + drct2 + "_" + str(j))
                    model.addConstr(T_m[d, t] - z_2[d, t, drct2, j] >= (1 - x["opaque", drct2, j]) * T_m_u,
                                    "lin4_z_2_" + timetag + "_" + drct2 + "_" + str(j))
            for i in range(len(U["window"])):
                # z_3[t, i] = x["window", i] * T_s
                model.addConstr(z_3[d, t, i] <= x["window", i] * T_s_o,
                                "lin1_z_3_" + timetag + "_" + str(i))
                model.addConstr(z_3[d, t, i] >= x["window", i] * T_s_u,
                                "lin2_z_3_" + timetag + "_" + str(i))
                model.addConstr(T_s[d, t] - z_3[d, t, i] <= (1 - x["window", i]) * T_s_o,
                                "lin3_z_3_" + timetag + "_" + str(i))
                model.addConstr(T_s[d, t] - z_3[d, t, i] >= (1 - x["window", i]) * T_s_u,
                                "lin4_z_3_" + timetag + "_" + str(i))
            for l in range(len(f_class["Am"])):
                # z_6[t, l] = x["class", l] * T_m
                model.addConstr(z_6[d, t, l] <= x["class", l] * T_m_o)
                model.addConstr(z_6[d, t, l] >= x["class", l] * T_m_u)
                model.addConstr(T_m[d, t] - z_6[d, t, l] <= (1 - x["class", l]) * T_m_o)
                model.addConstr(T_m[d, t] - z_6[d, t, l] >= (1 - x["class", l]) * T_m_u)
                # z_7[t, l] = x["class", l] * T_s
                model.addConstr(z_7[d, t, l] <= x["class", l] * T_s_o)
                model.addConstr(z_7[d, t, l] >= x["class", l] * T_s_u)
                model.addConstr(T_s[d, t] - z_7[d, t, l] <= (1 - x["class", l]) * T_s_o)
                model.addConstr(T_s[d, t] - z_7[d, t, l] >= (1 - x["class", l]) * T_s_u)

    for drct2 in direction2:
        for j in range(len(U["opaque", drct2])):
            for i in range(len(U["window"])):
                # z_1[drct2, j, i] = x["window", i]*x["opaque", drct2, j]
                model.addConstr(z_1[drct2, j, i] <= x["window", i],
                                "lin1_z1_" + drct2 + "_" + str(j) + "_" + str(i))
                model.addConstr(z_1[drct2, j, i] <= x["opaque", drct2, j],
                                "lin2_z1_" + drct2 + "_" + str(j) + "_" + str(i))
                model.addConstr(z_1[drct2, j, i] >= x["window", i] + x["opaque", drct2, j] - 1,
                                "lin3_z1_" + drct2 + "_" + str(j) + "_" + str(i))
            for l in range(len(f_class["Am"])):
                # z_4[drct2, j, l] = x["class", l] * x["opaque", drct2, j]
                model.addConstr(z_4[drct2, j, l] <= x["class", l])
                model.addConstr(z_4[drct2, j, l] <= x["opaque", drct2, j])
                model.addConstr(z_4[drct2, j, l] >= x["class", l] + x["opaque", drct2, j] - 1)

    for i in range(len(U["window"])):
        for l in range(len(f_class["Am"])):
            # z_5[i, l] = x["class", l] * x["window", i]
            model.addConstr(z_5[i, l] <= x["class", l])
            model.addConstr(z_5[i, l] <= x["window", i])
            model.addConstr(z_5[i, l] >= x["class", l] + x["window", i] - 1)

    # A_m (DIN EN ISO 13790, section 12.3.1.2, page 81, table 12)
    A_m = sum(x["class", l] * f_class["Am"][l] * A["f"] for l in range(len(f_class["Am"])))

    # C_m (DIN EN ISO 13790, section 12.3.1.2, page 81, table 12)
    C_m = (sum(sum(x["opaque", "wall", j] * kappa["opaque", dev][j] * A["opaque", dev]
                   for dev in ("wall", "intWall", "ceiling", "intFloor"))
               for j in range(len(kappa["opaque", "wall"]))) +
           sum(sum(x["opaque", drct4, j] * kappa["opaque", drct4][j] * A["opaque", drct4]
                   for j in range(len(kappa["opaque", drct4])))
               for drct4 in direction4))

    # Selection of building class based on C_m (scaled to kWh/K instead of J/K)
    model.addConstr(1.0 / 3600000 * C_m / A["f"] <= x["class", 0] * 95000.0 / 3600000
                    + x["class", 1] * 137500.0 / 3600000
                    + x["class", 2] * 212500.0 / 3600000
                    + x["class", 3] * 315000.0 / 3600000
                    + x["class", 4] * 100000000.0 / 3600000)
    model.addConstr(1.0 / 3600000 * C_m / A["f"] >= x["class", 1] * 95000.0 / 3600000
                    + x["class", 2] * 137500.0 / 3600000
                    + x["class", 3] * 212500.0 / 3600000
                    + x["class", 4] * 315000.0 / 3600000)

    # H_tr_w (DIN EN ISO 13790, section 8.3.1, page 44, eq. 18)
    H_tr_w = A["window"] * sum(x["window", i] * U["window"][i] for i in range(len(U["window"])))

    # Reference variables to reduce code length
    A_j_k = {}
    B_i_k = {}

    for d in range(number_of_representative_periods):
        for t in range(length_of_cluster):
            for drct3 in direction3:
                for j in range(len(U["opaque", "wall"])):
                    A_j_k[d, t, drct3, j] = (U["opaque", "wall"][j]
                                             * R_se["opaque", "wall"][j]
                                             * A["opaque", drct3]
                                             * (alpha_Sc["opaque", "wall"][j]
                                                * I_sol[drct3][d, t] * 1000 - h_r["opaque", "wall"][j]
                                                * F_r[drct3] * Delta_theta_er))
            for drct4 in direction4:
                for j in range(len(U["opaque", drct4])):
                    A_j_k[d, t, drct4, j] = (U["opaque", drct4][j]
                                             * R_se["opaque", drct4][j] * A["opaque", drct4]
                                             * (alpha_Sc["opaque", drct4][j]
                                                * I_sol[drct4][d, t] * 1000 - h_r["opaque", drct4][j]
                                                * F_r[drct4] * Delta_theta_er))

            for drct in direction:
                for i in range(len(U["window"])):
                    B_i_k[d, t, drct, i] = A["window", drct] * (g_gl["window"][i]
                                                                * (1 - F_F) * I_sol["window", drct][d, t] * 1000
                                                                * F_sh_gl - R_se["window"][i]
                                                                * U["window"][i] * h_r["window"][i]
                                                                * Delta_theta_er * F_r[drct])

    phi_sol = {}
    phi_m = {}
    phi_st = {}
    H_tr_em = {}
    phi_int_special = []
    delete_me = -1
    for t in range(length_of_cluster):
        if t % 24 == 0 and t != 0:
            delete_me = 0
        else:
            delete_me += 1
        phi_int_special.append(phi_int[delete_me])

    phi_int_special = np.array(phi_int_special)
    phi_ia_special = 0.5 * phi_int_special
    for d in range(number_of_representative_periods):
        for t in range(length_of_cluster):
            # heat flow phi_sol [kW]
            # (DIN EN ISO 13790, section 11.3.2, page 67, eq. 43)
            phi_sol[d, t] = (sum(sum(x["opaque", "wall", j] * A_j_k[d, t, drct3, j]
                                     for j in range(len(U["opaque", "wall"])))
                                 for drct3 in direction3) +
                             sum(sum(x["opaque", drct4, j] * A_j_k[d, t, drct4, j]
                                     for j in range(len(U["opaque", drct4])))
                                 for drct4 in direction4) +
                             sum(sum(x["window", i] * B_i_k[d, t, drct, i]
                                     for i in range(len(U["window"])))
                                 for drct in direction))

            # heat flow phi_m [kW]
            # (DIN EN ISO 13790, section C2, page 110, eq. C.2)
            phi_m[d, t] = (A_m / A_tot * 0.5 * phi_int_special[t] +
                           1.0 / A_tot * A["f"] *
                           sum(sum(sum(z_4["wall", j, l] * f_class["Am"][l] * A_j_k[d, t, drct3, j]
                                       for j in range(len(U["opaque", "wall"])))
                                   for drct3 in direction3) +
                               sum(sum(z_4[drct4, j, l] * f_class["Am"][l] * A_j_k[d, t, drct4, j]
                                       for j in range(len(U["opaque", drct4])))
                                   for drct4 in direction4) +
                               sum(sum(z_5[i, l] * f_class["Am"][l] * B_i_k[d, t, drct, i]
                                       for i in range(len(U["window"])))
                                   for drct in direction)
                               for l in range(len(f_class["Am"]))))

            # heat flow phi_st [kW]
            # (DIN EN ISO 13790, section C2, page 110, eq. C.3)
            phi_st[d, t] = (0.5 * phi_int_special[t] + phi_sol[d, t] - phi_m[d, t] -
                            H_tr_w / 9.1 / A_tot * 0.5 * phi_int_special[t] -
                            1.0 / 9.1 / A_tot * A["window"] *
                            sum(sum(sum(z_1["wall", j, i] * U["window"][i] * A_j_k[d, t, drct3, j]
                                        for j in range(len(U["opaque", "wall"])))
                                    for drct3 in direction3) +
                                sum(sum(z_1[drct4, j, i] * U["window"][i] * A_j_k[d, t, drct4, j]
                                        for j in range(len(U["opaque", drct4])))
                                    for drct4 in direction4) +
                                sum(x["window", i] * U["window"][i] * B_i_k[d, t, drct, i]
                                    for drct in direction)
                                for i in range(len(U["window"]))))

            # thermal transmittance coefficient H_tr_em [W/K]
            # Simplification: H_tr_em = H_tr_op
            # (DIN EN ISO 13790, section 8.3, page 43)
            H_tr_em[d, t] = sum(A["opaque", drct2] *
                                sum(U["opaque", drct2][j] * x["opaque", drct2, j] * b_tr[drct2][d, t]
                                    for j in range(len(U["opaque", drct2])))
                                for drct2 in direction2)

    for d in range(number_of_representative_periods):
        for t in range(length_of_cluster):
            # reference variables to reduce code length
            H_tr_em_x_T_m = sum(A["opaque", drct2] *
                                sum(z_2[d, t, drct2, j] * U["opaque", drct2][j] * b_tr[drct2][d, t]
                                    for j in range(len(U["opaque", drct2])))
                                for drct2 in direction2)

            H_tr_w_x_T_s = A["window"] * sum(z_3[d, t, i] * U["window"][i] for i in range(len(U["window"])))

            C_m_x_T_m = (sum(sum(z_2[d, t, "wall", j] * kappa["opaque", dev][j] * A["opaque", dev]
                                 for dev in ("wall", "intWall", "ceiling", "intFloor"))
                             for j in range(len(kappa["opaque", "wall"]))) +
                         sum(sum(z_2[d, t, drct4, j] * kappa["opaque", drct4][j] * A["opaque", drct4]
                                 for j in range(len(kappa["opaque", drct4])))
                             for drct4 in direction4))

            H_tr_ms_x_T_m = sum(z_6[d, t, l] * f_class["Am"][l] * A["f"] * h_ms
                                for l in range(len(f_class["Am"])))

            H_tr_ms_x_T_s = sum(z_7[d, t, l] * f_class["Am"][l] * A["f"] * h_ms
                                for l in range(len(f_class["Am"])))

            # T_previous for heat transfer of capacity C =  C_m * (T_m[t] * T_m[t-1]) / dt
            if t == 0:
                C_m_x_T_prev = C_m * T_init  # Initialization
            else:
                C_m_x_T_prev = (sum(sum(z_2[d, t - 1, "wall", j] * kappa["opaque", dev][j] * A["opaque", dev]
                                        for dev in ("wall", "intWall", "ceiling", "intFloor"))
                                    for j in range(len(kappa["opaque", "wall"]))) +
                                sum(sum(z_2[d, t - 1, drct4, j] * kappa["opaque", drct4][j] * A["opaque", drct4]
                                        for j in range(len(kappa["opaque", drct4])))
                                    for drct4 in direction4))

            ###
            # Linear system of equations to determine T_i, T_s, T_m, Q_HC (in kW)
            # compare (Michalak - 2014 - The simple hourly method of EN ISO 13790)
            # Using T_i instead of T_op for temperature control

            model.addConstr(H_tr_em_x_T_m + H_tr_ms_x_T_m + C_m_x_T_m / (3600 * dt) - H_tr_ms_x_T_s ==
                            phi_m[d, t] + H_tr_em[d, t] * T_e[d, t] + C_m_x_T_prev / (3600 * dt),
                            "Ax=b_Gl(1)_" + str(d) + "_" + str(t))

            model.addConstr(-H_tr_ms_x_T_m + H_tr_ms_x_T_s + H_tr_is * T_s[d, t] + H_tr_w_x_T_s - H_tr_is * T_i[d, t] ==
                            phi_st[d, t] + H_tr_w * T_e[d, t],
                            "Ax=b_Gl(2)_" + str(d) + "_" + str(t))

            model.addConstr(-H_tr_is * T_s[d, t] + H_ve * T_i[d, t] + H_tr_is * T_i[d, t] - 1000 * Q_HC[d, t] ==
                            phi_ia_special[t] + H_ve * T_e[d, t],
                            "Ax=b_Gl(3)_" + str(d) + "_" + str(t))

            model.addConstr(T_i[d, t] >= T_set_min, "Ax=b_Gl(4)_" + str(d) + "_" + str(t))

    # 5R1C ISO 13790 End
    ###################

    # Prevent boiler+EH and CHP+EH systems
    # After continuous CHP
    model.addConstr(x["EH"] + sum(x["Boiler", i] for i in range(number_devices["Boiler"])) <= 1, "Boiler_or_EH")
    model.addConstr(x["EH"] + sum(x["CHP", i] for i in range(number_devices["CHP"])) <= 1, "CHP_or_EH")

    ###############################################################################
    # Set start values and branching priority

    if options["load_start_vals"]:
        with open(options["filename_start_vals"], "r") as fin:
            for line in fin:
                line_split = line.replace("\n", "").split("\t")
                # (model.getVarByName(line_split[0])).Start = int(line_split[1])



    for key in x.keys():  # Branch on investment variables first
        x[key].BranchPriority = 100

    # Scenario specific restrictions
    # PV scenario (large taken from status-quo scenario):

    if options["pv_scenario"]:
         model.addConstr(sum(x["CHP", i] for i in range(len(cap["CHP"]))) == 0)
         model.addConstr(sum(x["HP", i] for i in range(len(cap["HP"]))) == 0)
         model.addConstr(sum(x["HP", i] for i in range(len(cap["HP"]))) == 0)
         model.addConstr(sum(x["Battery", i] for i in range(len(cap["Battery"]))) == 0)
         model.addConstr(x["PV"] == 0)
         model.addConstr(x["TES", 0] == 1)
         model.addConstr(x["Boiler", 0] == 1)
         model.addConstr(x["EH"] == 0)
         model.addConstr(x["STC"] == 0)
         model.addConstr(x["opaque", "wall", 0] == 1)
         model.addConstr(x["opaque", "roof", 0] == 1)
         model.addConstr(x["window", 0] == 1)
    else:
         #    # EnEV restrictions
         if options["enev_restrictions"]:
             # Set limits
             limit_envelope_U = {"roof": 0.15,
                                 "wall": 0.17,
                                 "window": 1.1}

             for component in ("roof", "wall"):
                 for i in range(len(U["opaque", component])):
                     if U["opaque", component][i] > limit_envelope_U[component]:
                         model.addConstr(x["opaque", component, i] == 0)
             for i in range(len(U["window"])):
                 if U["window"][i] > limit_envelope_U["window"]:
                     model.addConstr(x["window", i] == 0)

    ###

    if options["status_quo_with_storage"]:
        model.addConstr(sum(x["CHP", i] for i in range(len(cap["CHP"]))) == 0)
        model.addConstr(x["HP"] == 0)
        #model.addConstr(x["TES", 0] == 1)
        model.addConstr(x["Boiler", 0] == 1)
        model.addConstr(x["EH"] == 0)
        model.addConstr(x["STC"] == 0)
        model.addConstr(x["opaque", "wall", 0] == 1)
        model.addConstr(x["opaque", "roof", 0] == 1)
        model.addConstr(x["window", 0] == 1)
        #model.addConstr(x["Battery"] == 1)
        model.addConstr(x["PV"] == 0)
        #model.addConstr(area["STC"] == 10)
        if options["Variable"] == "Battery":
            model.addConstr(x["TES", 0] == 1)
            model.addConstr(x["Battery"] == 1)
            model.addConstr(cap_design["Battery"] == options["total_storage"])


        elif options["Variable"] == "TES":
            model.addConstr(x["Battery"] == 0)
            model.addConstr(sum(x["TES", i] for i in range(len(cap["TES"]))) == 1)

            model.addConstr(cap_design["TES"] == options["total_storage"])


        elif options["Variable"] == "Battery & TES":
            model.addConstr(x["Battery"] == 1)
            model.addConstr(sum(x["TES", i] for i in range(len(cap["TES"]))) == 1)
            model.addConstr(cap_design["TES"] + cap_design["Battery"] == options["total_storage"])

    if options["envelope_runs"]:
        print(options["bes"])
        print([options["envelope"]])
        model.addConstr(sum(x["CHP", i] for i in range(len(cap["CHP"]))) == options["bes"]["CHP"])
        model.addConstr(x["HP"] == options["bes"]["HP"])
        model.addConstr(x["TES", 0] == options["bes"]["TES0"])
        #model.addConstr(sum(x["TES", i] for i in range(len(cap["TES"]))) == options["bes"]["TES"])
        model.addConstr(x["Boiler", 0] == options["bes"]["Boiler0"])
        model.addConstr(x["EH"] == options["bes"]["EH"])
        model.addConstr(x["STC"] == options["bes"]["STC"])
        model.addConstr(x["Battery"] == options["bes"]["Battery"])
        model.addConstr(x["PV"] == options["bes"]["PV"])
        #model.addConstr(cap_design["Battery"] == options["total_storage"])
        #model.addConstr(cap_design["TES"] == options["total_storage"])
        #model.addConstr(cap_design["TES"] + cap_design["Battery"] == options["total_storage"])
        chosen = {}
        for j in ("wall", "roof", "window"):
            chosen[j] = options["envelope"][j]
        model.addConstr(x["opaque", "wall", chosen["wall"]] == 1)  # options["envelope"]["wall"])
        model.addConstr(x["opaque", "roof", chosen["roof"]] == 1)  # options["envelope"]["roof"])
        model.addConstr(x["window", chosen["window"]] == 1)  # options["envelope"]["window"])



    # Gurobi parameters
    model.params.TimeLimit = 3600
    model.params.MIPFocus = 1
    model.params.MIPGap = 0.02

    model.write("ModelDetails.lp")
    model.optimize()

    if model.status == gp.GRB.OPTIMAL or model.status == gp.GRB.TIME_LIMIT:
        res_c_total = c_total.X
        # My Addition

        res_x = {dev: x[dev].X for dev in ["EH", "PV", "STC", "Battery", "Battery_small", "Battery_large", "HP"]}

        for dev in number_devices.keys():
            res_x[dev] = np.array([x[dev, i].X for i in range(number_devices[dev])])

        res_x["floor"] = x["opaque", "floor", 0].X
        res_x["roof"] = np.array([x["opaque", "roof", i].X for i in range(len(U["opaque", "roof"]))])
        res_x["wall"] = np.array([x["opaque", "wall", i].X for i in range(len(U["opaque", "wall"]))])
        res_x["window"] = np.array([x["window", i].X for i in range(len(U["window"]))])
        res_x["class"] = np.array([x["class", i].X for i in range(len(f_class["Am"]))])


        res_y = {}
        for dev in ("Boiler", "CHP",):
            for i in range(number_devices[dev]):
                res_y[dev, i] = np.array([[y[dev, d, t, i].X for t in range(length_of_cluster)]
                                          for d in range(number_of_representative_periods)])

        for dev in ("HP",):
            res_y[dev] = np.array([[y[dev, d, t].X for t in range(length_of_cluster)]
                                   for d in range(number_of_representative_periods)])

        res_c_inv = {dev: c_inv[dev].X for dev in c_inv.keys()}
        res_c_serv = {dev: c_serv[dev].X for dev in c_serv.keys()}
        res_c_dem = {dev: c_dem[dev].X for dev in c_dem.keys()}
        res_c_met = c_met.X
        res_cap_design = {dev: cap_design[dev].X for dev in cap_design.keys()}
        res_cap_design["STC"] = area["STC"].X

        res_meter = {dev: meter[dev].X for dev in meter.keys()}
        res_revenue = {dev: revenue[dev].X for dev in revenue.keys()}

        res_area = {dev: area[dev].X for dev in area.keys()}

        res_subsidy = subsidy.X
        res_limit_pv_battery = limit_pv_battery.X

        res_energy = {}
        res_power = {}
        res_heat = {}
        res_storage = {}
        for dev in ("TES", "Battery", "Battery_small", "Battery_large"):
            res_storage[dev] = np.array([[storage[dev, d, t].X for t in range(length_of_cluster)]
                                         for d in range(number_of_representative_periods)])

        for dev in ("CHP", "Boiler"):
            res_energy[dev] = np.array([[energy[dev, d, t].X for t in range(length_of_cluster)]
                                        for d in range(number_of_representative_periods)])

        for dev in ("Boiler", "CHP", "HP", "EH", "STC"):
            res_heat[dev] = np.array([[heat[dev, d, t].X for t in range(length_of_cluster)]
                                      for d in range(number_of_representative_periods)])

        for dev in ("PV", "CHP", "HP", "EH", "Import", "STC",):
            res_power[dev] = np.array([[power[dev, d, t].X for t in range(length_of_cluster)]
                                       for d in range(number_of_representative_periods)])

        for dev in ("PV", "CHP"):
            for method in ("use", "sell"):
                res_power[dev, method] = np.array([[power[dev, d, t, method].X for t in range(length_of_cluster)]
                                                   for d in range(number_of_representative_periods)])

        res_power["Battery"] = {}
        res_power["Battery_small"] = {}
        res_power["Battery_large"] = {}
        for dev in ("Battery", "Battery_small","Battery_large"):
            for state in ("charge", "discharge"):
                res_power[dev]["total", state] = np.array([[power[dev, d, t, state].X for t in range(length_of_cluster)]
                                                           for d in range(number_of_representative_periods)])

        res_q_max = {"HP": np.array([[q_max["HP", d, t].X for t in range(length_of_cluster)]
                                     for d in range(number_of_representative_periods)])}
        for i in range(number_devices["Boiler"]):
            res_q_max["Boiler", i] = np.array([[q_max["Boiler", d, t, i].X for t in range(length_of_cluster)]
                                               for d in range(number_of_representative_periods)])
        res_p_max = {}

        for i in range(number_devices["CHP"]):
            res_p_max["CHP", i] = np.array([[p_max["CHP", d, t, i].X for t in range(length_of_cluster)]
                                            for d in range(number_of_representative_periods)])
            res_q_max["CHP", i] = np.array([[q_max["CHP", d, t, i].X for t in range(length_of_cluster)]
                                            for d in range(number_of_representative_periods)])

        # 5R1C ISO 13790
        ###############
        res_Q_HC = np.array(
            [[Q_HC[d, t].X for t in range(length_of_cluster)] for d in range(number_of_representative_periods)])
        res_T_i = np.array(
            [[T_i[d, t].X for t in range(length_of_cluster)] for d in range(number_of_representative_periods)])

        A_U = {}
        A_kappa = {}
        A_inv = {}
        print("Envelope:")
        for drct2 in direction2:
            for i in range(len(U["opaque", drct2])):
                if x["opaque", drct2, i].X == 1:
                    print(drct2, ":", i)
                    A_U["opaque", drct2] = A["opaque", drct2] * U["opaque", drct2][i]
                    A_kappa["opaque", drct2] = A["opaque", drct2] * kappa["opaque", drct2][i]
                    A_inv["opaque", drct2] = A["opaque", drct2] * inv["opaque", drct2][i]
        for i in range(len(U["window"])):
            if x["window", i].X == 1:
                print("window:", i)
                A_U["window"] = A["window"] * U["window"][i]
                A_inv["window"] = A["window"] * inv["window"][i]

        print("Energy system:")
        for dev in number_devices.keys():
            for i in range(number_devices[dev]):

                if dev in ("TES", "Boiler", "CHP"):
                    if x[dev, i].X == 1:
                        print(dev, "=", cap_design[dev].X, "[KW] | m3")

                else:
                    if x[dev, i].X == 1:
                        print(dev, "=", cap[dev][i], "[kW] | m3")

        for dev in ["EH", "Battery", "HP", ]:
            if x[dev].X == 1:
                print(dev, "=", cap_design[dev].X, u"[kW] | kWh ")

        if x["PV"].X == 1:
            print("PV_area", "=", area["PV"].X, "[m2]")
            print("PV_cap", "=", cap_design["PV"].X, "[kW]")
        if x["STC"].X == 1:
            print("STC", "=", area["STC"].X, "[m2]")

        Q_HC_sum = 0
        for d in range(number_of_representative_periods):
            for t in range(length_of_cluster):
                Q_HC_sum += res_Q_HC[d, t] * weight_days[d]
        print("Q_HC_sum =", Q_HC_sum, "[kWh]")

        #Q_nHC = ((A_U["window"] + A_U["opaque", "wall"] + A_U["opaque", "roof"] +
        #          A_U["opaque", "floor"] * f_g1 * f_g2 * G_w +
        #          ventilationRate * c_p_air * rho_air * V / 3600) * (T_set_min - T_ne))
        #print("DesignHeatLoad =", Q_nHC, "[kW]")

        emission_total = emission.X / 1000
        print("Emissions =", emission_total, "t CO2 per year")

        print("Costs =", res_c_total, "Euro")

        emissions_of_gas = emissions_gas1.X / 1000
        emissions_of_pv = emissions_PV1.X / 1000
        emissions_of_grid = emissions_grid1.X / 1000
        emissions_of_lca = emissions_lca1.X / 1000
        print("Emissions GAS =", emissions_of_gas, "t CO2 per year")
        print("Emissions PV =", emissions_of_pv, "t CO2 per year")
        print("Emissions GRID =", emissions_of_grid, "t CO2 per year")
        print("Emissions LCA =", emissions_of_lca, "t CO2 per year")


        # if x["TES", 0].X == 1:
        #    print("TES CAPACITY", "=", cap_design["TES"].X, "[kW]")
        # if x["TES", 1].X == 1:
        #    print("TES CAPACITY", "=", cap_design["TES"].X, "[kW]")

        # My Addition
        #Cap_Batt = cap_design["Battery"].X
        #Cap_TES = cap_design["TES"].X

        # 5R1C ISO 13790 End
        ###################
    else:  # Error in optimization
        model.computeIIS()
        model.write("model.ilp")
        print('\nConstraints:')
        for c in model.getConstrs():
            if c.IISConstr:
                print('%s' % c.constrName)
        print('\nBounds:')
        for v in model.getVars():
            if v.IISLB > 0:
                print('Lower bound: %s' % v.VarName)
            elif v.IISUB > 0:
                print('Upper bound: %s' % v.VarName)

    # Save results
    with open(options["filename_results"], "wb") as fout:
        pickle.dump(res_c_total, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(model.MIPGap, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(model.Params.TimeLimit, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_x, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_y, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_c_inv, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_c_serv, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_c_dem, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_c_met, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_cap_design, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_meter, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_revenue, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_area, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_subsidy, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_limit_pv_battery, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_energy, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_power, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_heat, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_storage, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_q_max, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_p_max, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_Q_HC, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(res_T_i, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(A_U, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(A_kappa, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(A_inv, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(Q_HC_sum, fout, pickle.HIGHEST_PROTOCOL)
        #pickle.dump(Q_nHC, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(emission_total, fout, pickle.HIGHEST_PROTOCOL)
        pickle.dump(emission_max, fout, pickle.HIGHEST_PROTOCOL)

    if options["store_start_vals"]:
        with open(options["filename_start_vals"], "w") as fout:
            for var in model.getVars():
                if var.VType == "B":
                    fout.write(var.VarName + "\t" + str(int(var.X)) + "\n")

    return (res_c_total, emission_total, emissions_of_gas, emissions_of_pv, emissions_of_grid, emissions_of_lca,
            res_x, res_cap_design, res_power, res_storage["Battery"], clustered)




