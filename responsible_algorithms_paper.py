""" Demonstrating how to load data, do battery optimisations and run FoM analysis."""

import os
import json
import numpy as np
import pandas as pd
import copy
import csv
import matplotlib
# matplotlib.use('pdf')
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib import gridspec
from math import pi

# BSGIP specific tools
import sys
sys.path.append("../")
from core_tools import loaders, cleaners, objects, functions, FoMs, enomo_intermediate
from c3x_enomo.c3x.enomo.models import EnergyStorage, EnergySystem, Demand, Generation, Tariff, DispatchRequest, LocalTariff
from c3x_enomo.c3x.enomo.energy_optimiser import EnergyOptimiser, OptimiserObjectiveSet, LocalEnergyOptimiser



def financial(meter, import_tariff, export_tariff):
    """
    Evaluate the financial outcome for a customer.

    Args:
        meter  (): an individual connection point in kW

        import_tariff (pd.Series): Expects this to be in $/kWh.

        export_tariff (pd.Series): Expects this to be in $/kWh.
    """

    # split meter flows into import and export
    connection_point_import = np.copy(meter)
    connection_point_export = np.copy(meter)
    for j, e in enumerate(meter):
        if e >= 0:
            connection_point_export[j] = 0
        else:
            connection_point_import[j] = 0

    cost = np.sum(connection_point_import * import_tariff)
    cost += np.sum(connection_point_export * export_tariff)

    return cost

def LEM_financial(tariffs, E_cl, E_cb, E_gc, E_bc, E_bl, E_gb, E_gl):
    """
    tariffs: specifies tariffs to be applied to aggregation of customers.

    E_cl, E_cb, E_gc, E_bc, E_bl, E_gb, E_gl: energy flows between CES battery 
                                        and net customers (all positive). 
                                        Abbreviations stand for: connection_point, 
                                        load, generation, battery
    """

    # All tariffs are passed to financial function as the second variable
    # (designed for import tariffs) because all E_ arrays are all positive.
    customer_cost  = financial(E_cl, tariffs['re_import_tariff'], 0)
    customer_cost += financial(E_cl, tariffs['rt_import_tariff'], 0)
    customer_cost += financial(E_bl, tariffs['le_import_tariff'], 0)
    customer_cost += financial(E_bl, tariffs['lt_import_tariff'], 0)
    customer_cost -= financial(E_gc, tariffs['re_export_tariff'], 0)
    customer_cost += financial(E_gc, tariffs['rt_export_tariff'], 0)
    customer_cost -= financial(E_gb, tariffs['le_export_tariff'], 0)
    customer_cost += financial(E_gb, tariffs['lt_export_tariff'], 0)
    customer_cost += financial(E_gl, tariffs['lt_import_tariff'], 0)
    customer_cost += financial(E_gl, tariffs['lt_export_tariff'], 0)

    battery_cost  = financial(E_gb, tariffs['le_import_tariff'], 0)
    battery_cost += financial(E_gb, tariffs['lt_import_tariff'], 0)
    battery_cost -= financial(E_bl, tariffs['le_export_tariff'], 0)
    battery_cost += financial(E_bl, tariffs['lt_export_tariff'], 0)
    battery_cost += financial(E_cb, tariffs['re_import_tariff'], 0)
    battery_cost += financial(E_cb, tariffs['rt_import_tariff'], 0)
    battery_cost -= financial(E_bc, tariffs['re_export_tariff'], 0)
    battery_cost += financial(E_bc, tariffs['rt_export_tariff'], 0)

    network_cost = -financial(E_cl, tariffs['rt_import_tariff'], 0)
    network_cost -= financial(E_bl, tariffs['lt_import_tariff'], 0)
    network_cost -= financial(E_bl, tariffs['lt_export_tariff'], 0)
    network_cost -= financial(E_gb, tariffs['lt_import_tariff'], 0)
    network_cost -= financial(E_gb, tariffs['rt_import_tariff'], 0)
    network_cost -= financial(E_bc, tariffs['rt_export_tariff'], 0)
    network_cost -= financial(E_gl, tariffs['lt_import_tariff'], 0)
    network_cost -= financial(E_gl, tariffs['lt_export_tariff'], 0)

    return customer_cost, battery_cost, network_cost


def apply_tariffs(tariffs):
    local_tariff = LocalTariff()
    local_tariff.add_local_energy_tariff_profile_export(dict(enumerate(tariffs['le_export_tariff'])))
    local_tariff.add_local_energy_tariff_profile_import(dict(enumerate(tariffs['le_import_tariff'])))
    local_tariff.add_local_transport_tariff_profile_export(dict(enumerate(tariffs['lt_export_tariff'])))
    local_tariff.add_local_transport_tariff_profile_import(dict(enumerate(tariffs['lt_import_tariff'])))
    local_tariff.add_remote_energy_tariff_profile_export(dict(enumerate(tariffs['re_export_tariff'])))
    local_tariff.add_remote_energy_tariff_profile_import(dict(enumerate(tariffs['re_import_tariff'])))
    local_tariff.add_remote_transport_tariff_profile_export(dict(enumerate(tariffs['rt_export_tariff'])))
    local_tariff.add_remote_transport_tariff_profile_import(dict(enumerate(tariffs['rt_import_tariff'])))
    energy_system.add_local_tariff(local_tariff)





##################### Load and check data #####################
# Trim time window
start_time = datetime(2018,1,1,0,0)
end_time = datetime(2018,1,31,23,55)
month="January"
# start_time = datetime(2018,7,1,0,0)
# end_time = datetime(2018,7,31,23,55)
# month="July"

interval_in_seconds = 300 #seconds
interval_in_minutes = 5
intervals_in_hour = 60/interval_in_minutes
intervals_in_day = 24*intervals_in_hour

# Load electricity spot market data
spot_price = pd.read_csv('spot_price.csv')
spot_price = spot_price.set_index('time')
spot_price = spot_price['RRP']/1000 # convert $/MWh to $/kWh
spot_price = spot_price.truncate(before=start_time.timestamp(), after=end_time.timestamp())

# Resample tariff data to 5 minute intervals to match load data
spot_price = spot_price.to_frame()
spot_price['index'] = spot_price.index.values
spot_price.index = pd.to_datetime(spot_price.index, unit='s')
spot_price = spot_price.resample('5min').ffill()
spot_price.index = spot_price['index']
spot_price = spot_price.drop(columns=['index'])
spot_price = spot_price.squeeze()
spot_price = spot_price.rename('rate')
for i in range(interval_in_minutes):
    last_time = spot_price.index[-1]
    last_data = spot_price.iloc[-1]
    spot_price[last_time+interval_in_seconds] = last_data
spot_price.index = np.arange(spot_price.index[0],spot_price.index[-1]+interval_in_seconds,interval_in_seconds)
num_intervals = len(spot_price)


# Calculate the carbon intensity of electricity as generation sources vary
carbon_data = pd.read_csv('emissions_intensity_2018_SA.csv')
carbon_data = carbon_data.set_index('settlementdate')
carbon_data.index = pd.to_datetime(carbon_data.index)
carbon_data.sort_index(inplace=True)
carbon_data.index = (carbon_data.index.tz_convert(None) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
carbon_data = carbon_data.truncate(before=start_time.timestamp(), after=end_time.timestamp())

SA_sum_power = pd.read_csv('sum_power_2018_SA.csv') # in MW
SA_sum_power = SA_sum_power.set_index('settlementdate')
SA_sum_power.index = pd.to_datetime(SA_sum_power.index)
SA_sum_power.index = (SA_sum_power.index.tz_convert(None) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
SA_sum_power = SA_sum_power.truncate(before=start_time.timestamp(), after=end_time.timestamp())

# carbon_data['sum_emissions'] is in tCO2e/MWh weighted by each generator's power
carbon_intensity = (carbon_data['sum_emissions'] / SA_sum_power['sum'])
carbon_intensity.name = 'rate'
# Scale carbon price to be roughly comparable to tariffs (carbon data is in CO2 per MW)
carbon_price = 2 # $/tCO2e

# Load community's load and solar generation data
# Load and solar flowing through customers' connection points
net_import_no_battery_df = pd.read_csv('net_import_no_battery.csv')
net_import_no_battery_df = net_import_no_battery_df.set_index('time')
net_import_no_battery_df = net_import_no_battery_df.truncate(before=start_time.timestamp(), after=end_time.timestamp())
net_export_no_battery_df = pd.read_csv('net_export_no_battery.csv')
net_export_no_battery_df = net_export_no_battery_df.set_index('time')
net_export_no_battery_df = net_export_no_battery_df.truncate(before=start_time.timestamp(), after=end_time.timestamp())
# BTM solar generation and load
total_load = pd.read_csv('total_load.csv')
total_load = total_load.set_index('time')
total_load = total_load.truncate(before=start_time.timestamp(), after=end_time.timestamp())
total_load = total_load['kW'].sum()/intervals_in_hour
total_solar_generation = pd.read_csv('total_solar.csv')
total_solar_generation = total_solar_generation.set_index('time')
total_solar_generation = total_solar_generation.truncate(before=start_time.timestamp(), after=end_time.timestamp())
total_solar_generation = total_solar_generation['kW'].sum()/intervals_in_hour

net_load_no_battery = net_import_no_battery_df + net_export_no_battery_df
# # split load into import and export
# net_import_no_battery = np.copy(net_load_no_battery['kW'].values)
# net_export_no_battery = np.copy(net_load_no_battery['kW'].values)
# for j, e in enumerate(net_load_no_battery['kW'].values):
#     if e >= 0:
#         net_export_no_battery[j] = 0
#     else:
#         net_import_no_battery[j] = 0

net_import_no_battery = net_import_no_battery_df.kW
net_export_no_battery = net_export_no_battery_df.kW

peak_power_import_no_battery = max(net_import_no_battery)
peak_power_export_no_battery = min(net_export_no_battery)
sum_energy_import_no_battery = sum(net_import_no_battery)/intervals_in_hour
sum_energy_export_no_battery = sum(net_export_no_battery)/intervals_in_hour
carbon_emissions_no_battery = sum(net_import_no_battery*carbon_intensity)


energy_system = EnergySystem()
load = Demand()
load.add_demand_profile(net_import_no_battery)
pv = Generation()
pv.add_generation_profile(net_export_no_battery)

battery_capacity_kWh = 1000
c_rate = 2
battery_p_charge_max = battery_capacity_kWh/c_rate
battery_p_discharge_max = -battery_p_charge_max
throughput_cost = 0.032 #$/kWh cost (wear and tear) of cycling the NSB
battery = EnergyStorage(max_capacity=battery_capacity_kWh*intervals_in_hour, 
                        charging_power_limit=battery_p_charge_max*intervals_in_hour, 
                        discharging_power_limit=battery_p_discharge_max*intervals_in_hour, 
                        charging_efficiency=1, discharging_efficiency=1, 
                        depth_of_discharge_limit=0.0, throughput_cost=throughput_cost,
                        initial_state_of_charge=0)
energy_system.add_energy_storage(battery)


re_i_factor = 1 # multiply base tariff values
re_e_factor = 1 # set export energy tariffs relative to re import
rt_i_factor = 1 # multiply base tariff values
rt_e_factor = 1 # set export transport tariffs relative to rt import
le_i_factor = 1 # set local import energy tariffs relative to re import
le_e_factor = 1 # set local export energy tariffs relative to le import
lt_i_factor = 0.5 # set local import transport tariffs relative to rt import
lt_e_factor = 0.5 # set local export transport tariffs relative to lt import

# Remove time intervals with missing data
temp_df = pd.concat([net_load_no_battery, spot_price], axis=1)
temp_df.dropna(inplace=True)
spot_price = temp_df['rate']

base_network_charges = copy.deepcopy(spot_price)
base_network_charges.values[:] = 0.1 # 10c/kWh

tariffs = {}
tariffs['re_import_tariff'] = spot_price
tariffs['re_export_tariff'] = re_e_factor*tariffs['re_import_tariff']
tariffs['rt_import_tariff'] = rt_i_factor*base_network_charges
tariffs['rt_export_tariff'] = rt_e_factor*tariffs['rt_import_tariff']
tariffs['le_import_tariff'] = le_i_factor*tariffs['re_import_tariff']
tariffs['le_export_tariff'] = le_e_factor*tariffs['le_import_tariff']
tariffs['lt_import_tariff'] = lt_i_factor*tariffs['rt_import_tariff']
tariffs['lt_export_tariff'] = lt_e_factor*tariffs['lt_import_tariff']
apply_tariffs(tariffs)

energy_system.add_demand(load)
energy_system.add_generation(pv)
total_solar_capacity = 600.0

customer_cost_no_battery = sum(net_import_no_battery * (tariffs['re_import_tariff'] + tariffs['rt_import_tariff']) +
                           abs(net_export_no_battery) * (-1*tariffs['re_export_tariff'] + tariffs['rt_export_tariff']))
print('Customer cost ${:.2f}'.format(customer_cost_no_battery))



fig = plt.figure(figsize=(13.6, 6))
gs = gridspec.GridSpec(1, 1)
ax1 = plt.subplot(gs[0])
time_axis = np.arange(num_intervals)
l2, = ax1.plot(time_axis,net_import_no_battery)
l2, = ax1.plot(time_axis,net_export_no_battery)
# l2, = ax1.plot(time_axis,carbon_price*carbon_intensity)
# l2, = ax1.plot(time_axis,net_load_no_battery)
plt.show()




##################### Optimise battery charge/discharging #####################
print('\n------ Optimise battery for carbon emissions ------')
# Change tariffs to artificial values to drive carbon emissions objective
re_i_factor = 1 # multiply base tariff values
re_e_factor = 1 # set export energy tariffs relative to re import
rt_i_factor = 0 # multiply base tariff values
rt_e_factor = 0 # set export transport tariffs relative to rt import
le_i_factor = 1 # set local import energy tariffs relative to re import
le_e_factor = 1 # set local export energy tariffs relative to le import
lt_i_factor = 0 # set local import transport tariffs relative to rt import
lt_e_factor = 0 # set local export transport tariffs relative to lt import
tariffs = {}
tariffs['re_import_tariff'] = carbon_price*carbon_intensity
tariffs['re_export_tariff'] = re_e_factor*tariffs['re_import_tariff']
tariffs['rt_import_tariff'] = rt_i_factor*base_network_charges
tariffs['rt_export_tariff'] = rt_e_factor*tariffs['rt_import_tariff']
tariffs['le_import_tariff'] = le_i_factor*tariffs['re_import_tariff']
tariffs['le_export_tariff'] = le_e_factor*tariffs['le_import_tariff']
tariffs['lt_import_tariff'] = lt_i_factor*tariffs['rt_import_tariff']
tariffs['lt_export_tariff'] = lt_e_factor*tariffs['lt_import_tariff']
apply_tariffs(tariffs)

opt_objective = OptimiserObjectiveSet.LocalModels
optimiser = LocalEnergyOptimiser(interval_in_minutes, num_intervals, energy_system, opt_objective)

optimised_battery_values ={ 'storage_charge_grid' : optimiser.values('storage_charge_grid'),
                            'storage_charge_generation' : optimiser.values('storage_charge_generation'),
                            'storage_discharge_demand' : optimiser.values('storage_discharge_demand'),
                            'storage_discharge_grid' : optimiser.values('storage_discharge_grid'),
                            'storage_state_of_charge' : optimiser.values('storage_state_of_charge'),
                            'net_import' : optimiser.values('local_net_import'),
                            'net_export' : optimiser.values('local_net_export'),
                            'demand_transfer' : optimiser.values('local_demand_transfer')}

E_cb = optimised_battery_values['storage_charge_grid']
E_gb = optimised_battery_values['storage_charge_generation']
E_bl = abs(optimised_battery_values['storage_discharge_demand'])
E_bc = abs(optimised_battery_values['storage_discharge_grid'])
E_cl = abs(optimised_battery_values['net_import'])
E_gc = abs(optimised_battery_values['net_export'])
E_gl = abs(optimised_battery_values['demand_transfer'])

E_cb_total_kWh = sum(E_cb)/intervals_in_hour
E_gb_total_kWh = sum(E_gb)/intervals_in_hour
E_bl_total_kWh = sum(E_bl)/intervals_in_hour
E_bc_total_kWh = sum(E_bc)/intervals_in_hour
E_gc_total_kWh = sum(E_gc)/intervals_in_hour
E_cl_total_kWh = sum(E_cl)/intervals_in_hour
E_gl_total_kWh = sum(E_gl)/intervals_in_hour

net_import = E_cl
net_export = -1*(E_bc+E_gc)
PCC_peak_power_import = max(net_import)
PCC_peak_power_export = min(net_export)
print('Peak power import {:.2f} kW'.format(PCC_peak_power_import))
print('Peak power export {:.2f} kW'.format(PCC_peak_power_export))
print('Sum peak powers {:.2f} kW'.format(abs(PCC_peak_power_export)+PCC_peak_power_import))
PCC_sum_energy_import = sum(net_import)/intervals_in_hour
print('Net energy imported {:.2f} kWh'.format((PCC_sum_energy_import)))
PCC_sum_energy_export = sum(net_export)/intervals_in_hour
print('Net energy exported {:.2f} kWh'.format(PCC_sum_energy_export))


battery_action_carbon = E_cb + E_gb - E_bl - E_bc
net_load_battery_carbon = E_cl + E_cb - E_gc - E_bc
peak_power_import_battery_carbon = PCC_peak_power_import
peak_power_export_battery_carbon = PCC_peak_power_export
sum_energy_import_battery_carbon = PCC_sum_energy_import
sum_energy_export_battery_carbon = PCC_sum_energy_export

battery_cycles = (E_cb_total_kWh + E_gb_total_kWh)/battery_capacity_kWh
nu_days = len(E_cb)/(12*24)
battery_cycles_per_day = battery_cycles/nu_days
print('Battery cycles per day {:.2f}'.format((battery_cycles_per_day)))

self_consumption_carbon = 1 - PCC_sum_energy_export/total_solar_generation
print('Self consumption {:.2f}%'.format(100*float(self_consumption_carbon)))
self_sufficiency_carbon = float(1 - PCC_sum_energy_import/total_load)
print('Self sufficiency {:.2f}%'.format(100*self_sufficiency_carbon))

# Reset tariffs to make costs consistent with other scenarios
re_i_factor = 1 # multiply base tariff values
re_e_factor = 1 # set export energy tariffs relative to re import
rt_i_factor = 1 # multiply base tariff values
rt_e_factor = 1 # set export transport tariffs relative to rt import
le_i_factor = 1 # set local import energy tariffs relative to re import
le_e_factor = 1 # set local export energy tariffs relative to le import
lt_i_factor = 0.5 # set local import transport tariffs relative to rt import
lt_e_factor = 0.5 # set local export transport tariffs relative to lt import
tariffs = {}
tariffs['re_import_tariff'] = spot_price
tariffs['re_export_tariff'] = re_e_factor*tariffs['re_import_tariff']
tariffs['rt_import_tariff'] = rt_i_factor*base_network_charges
tariffs['rt_export_tariff'] = rt_e_factor*tariffs['rt_import_tariff']
tariffs['le_import_tariff'] = le_i_factor*tariffs['re_import_tariff']
tariffs['le_export_tariff'] = le_e_factor*tariffs['le_import_tariff']
tariffs['lt_import_tariff'] = lt_i_factor*tariffs['rt_import_tariff']
tariffs['lt_export_tariff'] = lt_e_factor*tariffs['lt_import_tariff']
apply_tariffs(tariffs)

customer_cost, battery_cost, network_cost = LEM_financial(tariffs,E_cl, E_cb, E_gc, E_bc, E_bl, E_gb, E_gl)
print('Customer cost ${:.2f}'.format(customer_cost))
print('Battery cost ${:.2f}'.format(battery_cost))
customer_cost_carbon = customer_cost
battery_cost_carbon = battery_cost
carbon_emissions_carbon = sum(net_load_battery_carbon*carbon_intensity)
carbon_savings_carbon = carbon_emissions_no_battery - carbon_emissions_carbon


##################### Optimise battery charge/discharging #####################
print('\n------ Optimise battery for Battery-profit------')
opt_objective = OptimiserObjectiveSet.LocalModelsThirdParty 
optimiser = LocalEnergyOptimiser(interval_in_minutes, num_intervals, energy_system, opt_objective)

optimised_battery_values ={ 'storage_charge_grid' : optimiser.values('storage_charge_grid'),
                            'storage_charge_generation' : optimiser.values('storage_charge_generation'),
                            'storage_discharge_demand' : optimiser.values('storage_discharge_demand'),
                            'storage_discharge_grid' : optimiser.values('storage_discharge_grid'),
                            'storage_state_of_charge' : optimiser.values('storage_state_of_charge'),
                            'net_import' : optimiser.values('local_net_import'),
                            'net_export' : optimiser.values('local_net_export'),
                            'demand_transfer' : optimiser.values('local_demand_transfer')}

E_cb = optimised_battery_values['storage_charge_grid']
E_gb = optimised_battery_values['storage_charge_generation']
E_bl = abs(optimised_battery_values['storage_discharge_demand'])
E_bc = abs(optimised_battery_values['storage_discharge_grid'])
E_cl = abs(optimised_battery_values['net_import'])
E_gc = abs(optimised_battery_values['net_export'])
E_gl = abs(optimised_battery_values['demand_transfer'])

E_cb_total_kWh = sum(E_cb)/intervals_in_hour
E_gb_total_kWh = sum(E_gb)/intervals_in_hour
E_bl_total_kWh = sum(E_bl)/intervals_in_hour
E_bc_total_kWh = sum(E_bc)/intervals_in_hour
E_gc_total_kWh = sum(E_gc)/intervals_in_hour
E_cl_total_kWh = sum(E_cl)/intervals_in_hour
E_gl_total_kWh = sum(E_gl)/intervals_in_hour

net_import = E_cl
net_export = -1*(E_bc+E_gc)
PCC_peak_power_import = max(net_import)
PCC_peak_power_export = min(net_export)
print('Peak power import {:.2f} kW'.format(PCC_peak_power_import))
print('Peak power export {:.2f} kW'.format(PCC_peak_power_export))
print('Sum peak powers {:.2f} kW'.format(abs(PCC_peak_power_export)+PCC_peak_power_import))
PCC_sum_energy_import = sum(net_import)/intervals_in_hour
print('Net energy imported {:.2f} kWh'.format((PCC_sum_energy_import)))
PCC_sum_energy_export = sum(net_export)/intervals_in_hour
print('Net energy exported {:.2f} kWh'.format(PCC_sum_energy_export))

battery_action_profit = E_cb + E_gb - E_bl - E_bc
net_load_battery_profit = E_cl + E_cb - E_gc - E_bc
peak_power_import_battery_profit = PCC_peak_power_import
peak_power_export_battery_profit = PCC_peak_power_export
sum_energy_import_battery_profit = PCC_sum_energy_import
sum_energy_export_battery_profit = PCC_sum_energy_export

battery_cycles = (E_cb_total_kWh + E_gb_total_kWh)/battery_capacity_kWh
nu_days = len(E_cb)/(12*24)
battery_cycles_per_day = battery_cycles/nu_days
print('Battery cycles per day {:.2f}'.format((battery_cycles_per_day)))

self_consumption_profit = 1 - PCC_sum_energy_export/(total_solar_generation.sum()/intervals_in_hour)
print('Self consumption {:.2f}%'.format(100*float(self_consumption_profit)))
self_sufficiency_profit = float(1 - PCC_sum_energy_import/(total_load.sum()/intervals_in_hour))
print('Self sufficiency {:.2f}%'.format(100*self_sufficiency_profit))

customer_cost, battery_cost, network_cost = LEM_financial(tariffs,E_cl, E_cb, E_gc, E_bc, E_bl, E_gb, E_gl)
print('Customer cost ${:.2f}'.format(customer_cost))
print('Battery cost ${:.2f}'.format(battery_cost))
customer_cost_profit = customer_cost
battery_cost_profit = battery_cost
carbon_emissions_profit = sum(net_load_battery_profit*carbon_intensity)
carbon_savings_profit = carbon_emissions_no_battery - carbon_emissions_profit


##################### Optimise battery charge/discharging #####################
print('\n------ Optimise battery for community cost ------')
opt_objective = OptimiserObjectiveSet.LocalModels
optimiser = LocalEnergyOptimiser(interval_in_minutes, num_intervals, energy_system, opt_objective)

optimised_battery_values ={ 'storage_charge_grid' : optimiser.values('storage_charge_grid'),
                            'storage_charge_generation' : optimiser.values('storage_charge_generation'),
                            'storage_discharge_demand' : optimiser.values('storage_discharge_demand'),
                            'storage_discharge_grid' : optimiser.values('storage_discharge_grid'),
                            'storage_state_of_charge' : optimiser.values('storage_state_of_charge'),
                            'net_import' : optimiser.values('local_net_import'),
                            'net_export' : optimiser.values('local_net_export'),
                            'demand_transfer' : optimiser.values('local_demand_transfer')}

E_cb = optimised_battery_values['storage_charge_grid']
E_gb = optimised_battery_values['storage_charge_generation']
E_bl = abs(optimised_battery_values['storage_discharge_demand'])
E_bc = abs(optimised_battery_values['storage_discharge_grid'])
E_cl = abs(optimised_battery_values['net_import'])
E_gc = abs(optimised_battery_values['net_export'])
E_gl = abs(optimised_battery_values['demand_transfer'])

E_cb_total_kWh = sum(E_cb)/intervals_in_hour
E_gb_total_kWh = sum(E_gb)/intervals_in_hour
E_bl_total_kWh = sum(E_bl)/intervals_in_hour
E_bc_total_kWh = sum(E_bc)/intervals_in_hour
E_gc_total_kWh = sum(E_gc)/intervals_in_hour
E_cl_total_kWh = sum(E_cl)/intervals_in_hour
E_gl_total_kWh = sum(E_gl)/intervals_in_hour

net_import = E_cl
net_export = -1*(E_bc+E_gc)
PCC_peak_power_import = max(net_import)
PCC_peak_power_export = min(net_export)
print('Peak power import {:.2f} kW'.format(PCC_peak_power_import))
print('Peak power export {:.2f} kW'.format(PCC_peak_power_export))
print('Sum peak powers {:.2f} kW'.format(abs(PCC_peak_power_export)+PCC_peak_power_import))
PCC_sum_energy_import = sum(net_import)/intervals_in_hour
print('Net energy imported {:.2f} kWh'.format((PCC_sum_energy_import)))
PCC_sum_energy_export = sum(net_export)/intervals_in_hour
print('Net energy exported {:.2f} kWh'.format(PCC_sum_energy_export))

battery_action_communal = E_cb + E_gb - E_bl - E_bc
net_load_battery_communal = E_cl + E_cb - E_gc - E_bc
peak_power_import_battery_communal = PCC_peak_power_import
peak_power_export_battery_communal = PCC_peak_power_export
sum_energy_import_battery_communal = PCC_sum_energy_import
sum_energy_export_battery_communal = PCC_sum_energy_export

battery_cycles = (E_cb_total_kWh + E_gb_total_kWh)/battery_capacity_kWh
nu_days = len(E_cb)/(12*24)
battery_cycles_per_day = battery_cycles/nu_days
print('Battery cycles per day {:.2f}'.format((battery_cycles_per_day)))

self_consumption_communal = 1 - PCC_sum_energy_export/(total_solar_generation.sum()/intervals_in_hour)
print('Self consumption {:.2f}%'.format(100*float(self_consumption_communal)))
self_sufficiency_communal = float(1 - PCC_sum_energy_import/(total_load.sum()/intervals_in_hour))
print('Self sufficiency {:.2f}%'.format(100*self_sufficiency_communal))

customer_cost, battery_cost, network_cost = LEM_financial(tariffs,E_cl, E_cb, E_gc, E_bc, E_bl, E_gb, E_gl)
print('Customer cost ${:.2f}'.format(customer_cost))
print('Battery cost ${:.2f}'.format(battery_cost))
customer_cost_communal = customer_cost
battery_cost_communal = battery_cost
carbon_emissions_communal = sum(net_load_battery_communal*carbon_intensity)
carbon_savings_communal = carbon_emissions_no_battery - carbon_emissions_communal


##################### Optimise battery charge/discharging #####################
print('\n------ Optimise battery for electrical conditions ------')
# Change tariffs to force battery to at all costs preference local supplies
re_i_factor = 1 # multiply base tariff values
re_e_factor = 1 # set export energy tariffs relative to re import
rt_i_factor = 1e4 # multiply base tariff values
rt_e_factor = 1e4 # set export transport tariffs relative to rt import
le_i_factor = 1 # set local import energy tariffs relative to re import
le_e_factor = 1 # set local export energy tariffs relative to le import
lt_i_factor = 0 # set local import transport tariffs relative to rt import
lt_e_factor = 0 # set local export transport tariffs relative to lt import
tariffs = {}
tariffs['re_import_tariff'] = spot_price
tariffs['re_export_tariff'] = re_e_factor*tariffs['re_import_tariff']
tariffs['rt_import_tariff'] = rt_i_factor*base_network_charges
tariffs['rt_export_tariff'] = rt_e_factor*tariffs['rt_import_tariff']
tariffs['le_import_tariff'] = le_i_factor*tariffs['re_import_tariff']
tariffs['le_export_tariff'] = le_e_factor*tariffs['le_import_tariff']
tariffs['lt_import_tariff'] = lt_i_factor*tariffs['rt_import_tariff']
tariffs['lt_export_tariff'] = lt_e_factor*tariffs['lt_import_tariff']
apply_tariffs(tariffs)


opt_objective = OptimiserObjectiveSet.LocalModels
opt_objective += OptimiserObjectiveSet.LocalPeakOptimisation
optimiser = LocalEnergyOptimiser(interval_in_minutes, num_intervals, energy_system, opt_objective)

optimised_battery_values ={ 'storage_charge_grid' : optimiser.values('storage_charge_grid'),
                            'storage_charge_generation' : optimiser.values('storage_charge_generation'),
                            'storage_discharge_demand' : optimiser.values('storage_discharge_demand'),
                            'storage_discharge_grid' : optimiser.values('storage_discharge_grid'),
                            'storage_state_of_charge' : optimiser.values('storage_state_of_charge'),
                            'net_import' : optimiser.values('local_net_import'),
                            'net_export' : optimiser.values('local_net_export'),
                            'demand_transfer' : optimiser.values('local_demand_transfer')}

E_cb = optimised_battery_values['storage_charge_grid']
E_gb = optimised_battery_values['storage_charge_generation']
E_bl = abs(optimised_battery_values['storage_discharge_demand'])
E_bc = abs(optimised_battery_values['storage_discharge_grid'])
E_cl = abs(optimised_battery_values['net_import'])
E_gc = abs(optimised_battery_values['net_export'])
E_gl = abs(optimised_battery_values['demand_transfer'])

E_cb_total_kWh = sum(E_cb)/intervals_in_hour
E_gb_total_kWh = sum(E_gb)/intervals_in_hour
E_bl_total_kWh = sum(E_bl)/intervals_in_hour
E_bc_total_kWh = sum(E_bc)/intervals_in_hour
E_gc_total_kWh = sum(E_gc)/intervals_in_hour
E_cl_total_kWh = sum(E_cl)/intervals_in_hour
E_gl_total_kWh = sum(E_gl)/intervals_in_hour

net_import = E_cl
net_export = -1*(E_bc+E_gc)
PCC_peak_power_import = max(net_import)
PCC_peak_power_export = min(net_export)
print('Peak power import {:.2f} kW'.format(PCC_peak_power_import))
print('Peak power export {:.2f} kW'.format(PCC_peak_power_export))
print('Sum peak powers {:.2f} kW'.format(abs(PCC_peak_power_export)+PCC_peak_power_import))
PCC_sum_energy_import = sum(net_import)/intervals_in_hour
print('Net energy imported {:.2f} kWh'.format((PCC_sum_energy_import)))
PCC_sum_energy_export = sum(net_export)/intervals_in_hour
print('Net energy exported {:.2f} kWh'.format(PCC_sum_energy_export))

battery_action_sufficiency = E_cb + E_gb - E_bl - E_bc
net_load_battery_sufficiency = E_cl + E_cb - E_gc - E_bc
peak_power_import_battery_sufficiency = PCC_peak_power_import
peak_power_export_battery_sufficiency = PCC_peak_power_export
sum_energy_import_battery_sufficiency = PCC_sum_energy_import
sum_energy_export_battery_sufficiency = PCC_sum_energy_export

battery_cycles = (E_cb_total_kWh + E_gb_total_kWh)/battery_capacity_kWh
nu_days = len(E_cb)/(12*24)
battery_cycles_per_day = battery_cycles/nu_days
print('Battery cycles per day {:.2f}'.format((battery_cycles_per_day)))

self_consumption_sufficiency = 1 - PCC_sum_energy_export/total_solar_generation.sum()/intervals_in_hour
print('Self consumption {:.2f}%'.format(100*float(self_consumption_sufficiency)))
self_sufficiency_sufficiency = float(1 - PCC_sum_energy_import/total_load.sum()/intervals_in_hour)
print('Self sufficiency {:.2f}%'.format(100*self_sufficiency_sufficiency))


# Reset tariffs to make costs consistent with other scenarios
re_i_factor = 1 # multiply base tariff values
re_e_factor = 1 # set export energy tariffs relative to re import
rt_i_factor = 1 # multiply base tariff values
rt_e_factor = 1 # set export transport tariffs relative to rt import
le_i_factor = 1 # set local import energy tariffs relative to re import
le_e_factor = 1 # set local export energy tariffs relative to le import
lt_i_factor = 0.5 # set local import transport tariffs relative to rt import
lt_e_factor = 0.5 # set local export transport tariffs relative to lt import
tariffs = {}
tariffs['re_import_tariff'] = spot_price
tariffs['re_export_tariff'] = re_e_factor*tariffs['re_import_tariff']
tariffs['rt_import_tariff'] = rt_i_factor*base_network_charges
tariffs['rt_export_tariff'] = rt_e_factor*tariffs['rt_import_tariff']
tariffs['le_import_tariff'] = le_i_factor*tariffs['re_import_tariff']
tariffs['le_export_tariff'] = le_e_factor*tariffs['le_import_tariff']
tariffs['lt_import_tariff'] = lt_i_factor*tariffs['rt_import_tariff']
tariffs['lt_export_tariff'] = lt_e_factor*tariffs['lt_import_tariff']
apply_tariffs(tariffs)

customer_cost, battery_cost, network_cost = LEM_financial(tariffs,E_cl, E_cb, E_gc, E_bc, E_bl, E_gb, E_gl)
print('Customer cost ${:.2f}'.format(customer_cost))
print('Battery cost ${:.2f}'.format(battery_cost))
customer_cost_sufficiency = customer_cost
battery_cost_sufficiency = battery_cost
carbon_emissions_sufficiency = sum(net_load_battery_sufficiency*carbon_intensity)
carbon_savings_sufficiency = carbon_emissions_no_battery - carbon_emissions_sufficiency


##################### Optimise battery charge/discharging #####################
print('\n------ Coptimise battery for carbon and profit ------')

# Change tariffs to force battery to at all costs preference local supplies
carbon_weighting = 0.5 # 0.5 = weighting carbon and spot price equally
re_i_factor = 1 # multiply base tariff values
re_e_factor = 1 # set export energy tariffs relative to re import
rt_i_factor = 1 # multiply base tariff values
rt_e_factor = 1 # set export transport tariffs relative to rt import
le_i_factor = 1 # zero carbon cost
le_e_factor = 1 # earn carbon price at regular grid rates
lt_i_factor = 0.5 # set local import transport tariffs relative to rt import
lt_e_factor = 0.5 # set local export transport tariffs relative to lt import
tariffs = {}
tariffs['re_import_tariff'] = carbon_weighting*carbon_price*carbon_intensity + (1-carbon_weighting)*spot_price
tariffs['re_export_tariff'] = re_e_factor*tariffs['re_import_tariff']
tariffs['rt_import_tariff'] = rt_i_factor*base_network_charges
tariffs['rt_export_tariff'] = rt_e_factor*tariffs['rt_import_tariff']
tariffs['le_import_tariff'] = le_i_factor*tariffs['re_import_tariff']
tariffs['le_export_tariff'] = le_e_factor*tariffs['le_import_tariff']
tariffs['lt_import_tariff'] = lt_i_factor*tariffs['rt_import_tariff']
tariffs['lt_export_tariff'] = lt_e_factor*tariffs['lt_import_tariff']
apply_tariffs(tariffs)

opt_objective = OptimiserObjectiveSet.LocalModelsThirdParty
optimiser = LocalEnergyOptimiser(interval_in_minutes, num_intervals, energy_system, opt_objective)

optimised_battery_values ={ 'storage_charge_grid' : optimiser.values('storage_charge_grid'),
                            'storage_charge_generation' : optimiser.values('storage_charge_generation'),
                            'storage_discharge_demand' : optimiser.values('storage_discharge_demand'),
                            'storage_discharge_grid' : optimiser.values('storage_discharge_grid'),
                            'storage_state_of_charge' : optimiser.values('storage_state_of_charge'),
                            'net_import' : optimiser.values('local_net_import'),
                            'net_export' : optimiser.values('local_net_export'),
                            'demand_transfer' : optimiser.values('local_demand_transfer')}

E_cb = optimised_battery_values['storage_charge_grid']
E_gb = optimised_battery_values['storage_charge_generation']
E_bl = abs(optimised_battery_values['storage_discharge_demand'])
E_bc = abs(optimised_battery_values['storage_discharge_grid'])
E_cl = abs(optimised_battery_values['net_import'])
E_gc = abs(optimised_battery_values['net_export'])
E_gl = abs(optimised_battery_values['demand_transfer'])

E_cb_total_kWh = sum(E_cb)/intervals_in_hour
E_gb_total_kWh = sum(E_gb)/intervals_in_hour
E_bl_total_kWh = sum(E_bl)/intervals_in_hour
E_bc_total_kWh = sum(E_bc)/intervals_in_hour
E_gc_total_kWh = sum(E_gc)/intervals_in_hour
E_cl_total_kWh = sum(E_cl)/intervals_in_hour
E_gl_total_kWh = sum(E_gl)/intervals_in_hour

net_import = E_cl
net_export = -1*(E_bc+E_gc)
PCC_peak_power_import = max(net_import)
PCC_peak_power_export = min(net_export)
print('Peak power import {:.2f} kW'.format(PCC_peak_power_import))
print('Peak power export {:.2f} kW'.format(PCC_peak_power_export))
print('Sum peak powers {:.2f} kW'.format(abs(PCC_peak_power_export)+PCC_peak_power_import))
PCC_sum_energy_import = sum(net_import)/intervals_in_hour
print('Net energy imported {:.2f} kWh'.format((PCC_sum_energy_import)))
PCC_sum_energy_export = sum(net_export)/intervals_in_hour
print('Net energy exported {:.2f} kWh'.format(PCC_sum_energy_export))

battery_action_carbon_coopt = E_cb + E_gb - E_bl - E_bc
net_load_battery_carbon_coopt = E_cl + E_cb - E_gc - E_bc
peak_power_import_battery_carbon_coopt = PCC_peak_power_import
peak_power_export_battery_carbon_coopt = PCC_peak_power_export
sum_energy_import_battery_carbon_coopt = PCC_sum_energy_import
sum_energy_export_battery_carbon_coopt = PCC_sum_energy_export

battery_cycles = (E_cb_total_kWh + E_gb_total_kWh)/battery_capacity_kWh
nu_days = len(E_cb)/(12*24)
battery_cycles_per_day = battery_cycles/nu_days
print('Battery cycles per day {:.2f}'.format((battery_cycles_per_day)))

self_consumption_carbon_coopt = 1 - PCC_sum_energy_export/(total_solar_generation.sum()/intervals_in_hour)
print('Self consumption {:.2f}%'.format(100*float(self_consumption_carbon_coopt)))
self_sufficiency_carbon_coopt = float(1 - PCC_sum_energy_import/(total_load.sum()/intervals_in_hour))
print('Self sufficiency {:.2f}%'.format(100*self_sufficiency_carbon_coopt))


# Reset tariffs to make costs consistent with other scenarios
re_i_factor = 1 # multiply base tariff values
re_e_factor = 1 # set export energy tariffs relative to re import
rt_i_factor = 1 # multiply base tariff values
rt_e_factor = 1 # set export transport tariffs relative to rt import
le_i_factor = 1 # set local import energy tariffs relative to re import
le_e_factor = 1 # set local export energy tariffs relative to le import
lt_i_factor = 0.5 # set local import transport tariffs relative to rt import
lt_e_factor = 0.5 # set local export transport tariffs relative to lt import
tariffs = {}
tariffs['re_import_tariff'] = spot_price
tariffs['re_export_tariff'] = re_e_factor*tariffs['re_import_tariff']
tariffs['rt_import_tariff'] = rt_i_factor*base_network_charges
tariffs['rt_export_tariff'] = rt_e_factor*tariffs['rt_import_tariff']
tariffs['le_import_tariff'] = le_i_factor*tariffs['re_import_tariff']
tariffs['le_export_tariff'] = le_e_factor*tariffs['le_import_tariff']
tariffs['lt_import_tariff'] = lt_i_factor*tariffs['rt_import_tariff']
tariffs['lt_export_tariff'] = lt_e_factor*tariffs['lt_import_tariff']
apply_tariffs(tariffs)

customer_cost, battery_cost, network_cost = LEM_financial(tariffs,E_cl, E_cb, E_gc, E_bc, E_bl, E_gb, E_gl)
print('Customer cost ${:.2f}'.format(customer_cost))
print('Battery cost ${:.2f}'.format(battery_cost))
customer_cost_carbon_coopt = customer_cost
battery_cost_carbon_coopt = battery_cost
carbon_emissions_carbon_coopt = sum(net_load_battery_carbon_coopt*carbon_intensity)
carbon_savings_carbon_coopt = carbon_emissions_no_battery - carbon_emissions_carbon_coopt



##################### Unoptimised battery charge/discharging #####################
print('\n------ Run battery under Time of Use pattern ------')
# Charge NSB 6-18:00 discharge outside this
ToU_charge_length = 12 # hrs
ToU_discharge_length = 12 # hrs
ToU_charge_rate = battery_capacity_kWh/ToU_charge_length
ToU_discharge_rate = -battery_capacity_kWh/ToU_discharge_length

ToU_battery = np.zeros(num_intervals)
for i in range(num_intervals):
    if i%intervals_in_day <intervals_in_hour*6:
        ToU_battery[i] = ToU_discharge_rate
    elif intervals_in_hour*6 <= i%intervals_in_day < intervals_in_hour*18:
        ToU_battery[i] = ToU_charge_rate
    elif i%intervals_in_day >= intervals_in_hour*18:
        ToU_battery[i] = ToU_discharge_rate

ToU_battery_series = pd.Series(index=spot_price.index, data=ToU_battery)

net_load_no_battery_imports = np.where(net_load_no_battery['kW'].values > 0, net_load_no_battery['kW'].values, 0)
net_load_no_battery_exports = np.where(net_load_no_battery['kW'].values < 0, net_load_no_battery['kW'].values, 0)

net_load_w_ToU_battery = net_load_no_battery['kW'].values + ToU_battery
net_load_w_ToU_battery_imports = np.where(net_load_w_ToU_battery > 0, net_load_w_ToU_battery, 0)
net_load_w_ToU_battery_exports = np.where(net_load_w_ToU_battery < 0, net_load_w_ToU_battery, 0)


ToU_battery_charging = np.where(ToU_battery > 0, ToU_battery, 0)
ToU_battery_discharging = np.where(ToU_battery < 0, ToU_battery, 0)
E_gl = household_total_exports['kW'].values - net_load_no_battery_exports
excess_solar = household_total_exports['kW'].values - E_gl

E_gb = np.zeros(len(net_load_no_battery_imports))
E_gc = np.zeros(len(net_load_no_battery_imports))
for i, e in enumerate(ToU_battery):
    if ToU_battery_charging[i] < excess_solar[i]:
        E_gb[i] = ToU_battery_charging[i]
        E_gc[i] = excess_solar[i] - ToU_battery_charging[i]
    elif 0 < excess_solar[i] and excess_solar[i] < ToU_battery_charging[i]:
        E_gb[i] = excess_solar[i]

E_cb = ToU_battery_charging - E_gb
E_bc = net_load_w_ToU_battery_exports - E_gc
E_cl = net_load_w_ToU_battery_imports - E_cb
E_bl = ToU_battery_discharging - E_bc

E_cb_total_kWh = sum(E_cb)/intervals_in_hour
E_gb_total_kWh = sum(E_gb)/intervals_in_hour
E_bl_total_kWh = sum(E_bl)/intervals_in_hour
E_bc_total_kWh = sum(E_bc)/intervals_in_hour
E_gc_total_kWh = sum(E_gc)/intervals_in_hour
E_cl_total_kWh = sum(E_cl)/intervals_in_hour
E_gl_total_kWh = sum(E_gl)/intervals_in_hour

battery_action_ToU = E_cb + E_gb - E_bl - E_bc
net_load_battery_ToU = E_cl + E_cb - E_gc - E_bc
peak_power_import_battery_ToU = max(net_load_w_ToU_battery_imports)
peak_power_export_battery_ToU = min(net_load_w_ToU_battery_exports)
sum_energy_import_battery_ToU = sum(net_load_w_ToU_battery_imports)/intervals_in_hour
sum_energy_export_battery_ToU = sum(net_load_w_ToU_battery_exports)/intervals_in_hour

battery_cycles = (E_cb_total_kWh + E_gb_total_kWh)/battery_capacity_kWh
nu_days = len(E_cb)/(12*24)
battery_cycles_per_day = battery_cycles/nu_days
print('Battery cycles per day {:.2f}'.format((battery_cycles_per_day)))

self_consumption_ToU = 1 - sum_energy_export_battery_ToU/total_solar_generation['kW'].values.sum()
print('Self consumption {:.2f}%'.format(100*float(self_consumption_ToU)))
self_sufficiency_ToU = float(1 - sum_energy_import_battery_ToU/total_load['kW'].values.sum())
print('Self sufficiency {:.2f}%'.format(100*self_sufficiency_ToU))


# Reset tariffs to make costs consistent with other scenarios
re_i_factor = 1 # multiply base tariff values
re_e_factor = 1 # set export energy tariffs relative to re import
rt_i_factor = 1 # multiply base tariff values
rt_e_factor = 1 # set export transport tariffs relative to rt import
le_i_factor = 1 # set local import energy tariffs relative to re import
le_e_factor = 1 # set local export energy tariffs relative to le import
lt_i_factor = 0.5 # set local import transport tariffs relative to rt import
lt_e_factor = 0.5 # set local export transport tariffs relative to lt import
tariffs = {}
tariffs['re_import_tariff'] = spot_price
tariffs['re_export_tariff'] = re_e_factor*tariffs['re_import_tariff']
tariffs['rt_import_tariff'] = rt_i_factor*base_network_charges
tariffs['rt_export_tariff'] = rt_e_factor*tariffs['rt_import_tariff']
tariffs['le_import_tariff'] = le_i_factor*tariffs['re_import_tariff']
tariffs['le_export_tariff'] = le_e_factor*tariffs['le_import_tariff']
tariffs['lt_import_tariff'] = lt_i_factor*tariffs['rt_import_tariff']
tariffs['lt_export_tariff'] = lt_e_factor*tariffs['lt_import_tariff']
apply_tariffs(tariffs)

customer_cost, battery_cost, network_cost = LEM_financial(tariffs,E_cl, E_cb, E_gc, E_bc, E_bl, E_gb, E_gl)
print('Customer cost ${:.2f}'.format(customer_cost))
print('Battery cost ${:.2f}'.format(battery_cost))
customer_cost_ToU = customer_cost
battery_cost_ToU = battery_cost
carbon_emissions_ToU = sum(net_load_battery_ToU*carbon_intensity)
carbon_savings_ToU = carbon_emissions_no_battery - carbon_emissions_ToU






#### Plot results ####
time_axis = np.arange(num_intervals)

## Plot net load without NSB
fig = plt.figure(figsize=(13.6, 6))
gs = gridspec.GridSpec(1, 1)

start_day = 19
end_day = 21
xticks = time_axis[start_day*288:end_day*288][::12*6]
xticks = np.append(xticks,time_axis[start_day*288:end_day*288][-1])

ax1 = plt.subplot(gs[0])
ax1.axhline(y=0, linewidth=2, color='k')
l2, = ax1.plot(time_axis[start_day*288:end_day*288], net_load_no_battery['kW'][start_day*288:end_day*288],  
    linestyle='-', marker='', markersize=16, markevery=10, label='No NSB')
ax1.set_ylabel('Net load (kW)')
ax1.set_xlim([time_axis[start_day*288:end_day*288][0], time_axis[start_day*288:end_day*288][-1]])

ax1.set_xticks(xticks)
ax1.set_xticklabels(['0','6','12','18','0','6','13','18','0'])
ax1.set_xlabel('Time of day')
ax1.axvspan(time_axis[start_day*288+int(12*7.75)],time_axis[start_day*288+12*17], 
    alpha=0.2, color='purple')
ax1.axvspan(time_axis[start_day*288+int(12*(7.75+24))],time_axis[start_day*288+12*(24+17)], 
    alpha=0.2, color='purple')
ax1.set_yticks([200,0,-200])
ax1.set_yticks([-200,-100,0,100,200])
ax1.grid(False)
plt.savefig('net_load-{0}kW_c{1}-{2}'.format(battery_capacity_kWh, c_rate, month))



## Plot net load and NSB actions under financial algorithms
fig = plt.figure(figsize=(13.6, 12))
gs = gridspec.GridSpec(3, 1, height_ratios=[1,1,1])

ax0 = plt.subplot(gs[0])
l2, = ax0.plot(time_axis[start_day*288:end_day*288], spot_price[start_day*288:end_day*288], 
    linestyle='-', label='Spot price')
ax0.set_ylabel('Price ($/kWh)')
ax0.set_xlim([time_axis[start_day*288:end_day*288][0], time_axis[start_day*288:end_day*288][-1]])
ax0.set_xticks(time_axis[start_day*288:end_day*288][::12*6])
ax0.set_ylim([0, 1])
ax0.set_yticks([0,.5,1])
ax0.set_xticklabels([])
ax0.axvspan(time_axis[start_day*288+int(12*7.75)],time_axis[start_day*288+12*17], 
    alpha=0.2, color='purple')
ax0.axvspan(time_axis[start_day*288+int(12*(7.75+24))],time_axis[start_day*288+12*(24+17)], 
    alpha=0.2, color='purple')
ax0.grid(False)
ax0.set_yticks([0,0.25,0.5,0.75,1])

ax2 = plt.subplot(gs[1])
ax2.axhline(y=0, linewidth=2, color='k')
l1, = ax2.plot(time_axis, battery_action_profit,  
    linestyle='-', markersize=16, markevery=10, label='Battery-profit')
l2, = ax2.plot(time_axis, battery_action_communal, 
    linestyle='--', marker='^', markersize=16, markevery=10, label='Communal-savings')
ax2.set_ylabel('NSB action (kW)')
ax2.set_xticks(time_axis[start_day*288:end_day*288][::12*6])
ax2.set_xlim([time_axis[start_day*288:end_day*288][0], time_axis[start_day*288:end_day*288][-1]])
ax2.set_xticklabels([])
ax2.set_ylim([-200, 200])
ax2.axvspan(time_axis[start_day*288+int(12*7.75)],time_axis[start_day*288+12*17], 
    alpha=0.2, color='purple')
ax2.axvspan(time_axis[start_day*288+int(12*(7.75+24))],time_axis[start_day*288+12*(24+17)], 
    alpha=0.2, color='purple')
ax2.legend(ncol=2, fontsize=20)
ax2.grid(False)
ax2.set_yticks([-200,-100,0,100,200])

ax4 = plt.subplot(gs[2])
ax4.axhline(y=0, linewidth=2, color='k')
l1, = ax4.plot(time_axis, net_load_no_battery,  
    linestyle='-', markersize=16, markevery=10, label='No NSB')
l1, = ax4.plot(time_axis, net_load_battery_profit,  
    linestyle='-', markersize=16, markevery=10, label='Battery-profit')
l2, = ax4.plot(time_axis, net_load_battery_communal,  
    linestyle='--', marker='^', markersize=16, markevery=10, label='Communal-savings')
ax4.set_ylabel('Net load (kW)')
ax4.set_xticks(time_axis[start_day*288:end_day*288][::12*6])
ax4.set_xlim([time_axis[start_day*288:end_day*288][0], time_axis[start_day*288:end_day*288][-1]])
ax4.axvspan(time_axis[start_day*288+int(12*7.75)],time_axis[start_day*288+12*17], 
    alpha=0.2, color='purple')
ax4.axvspan(time_axis[start_day*288+int(12*(7.75+24))],time_axis[start_day*288+12*(24+17)], 
    alpha=0.2, color='purple')
ax4.legend(ncol=3, fontsize=20, loc='upper center')
ax4.grid(False)
ax4.set_yticks([-750,-500,-250,0,250,500])

ax4.set_xticks(xticks)
ax4.set_xticklabels(['0','6','12','18','0','6','13','18','0'])
ax4.set_xlabel('Time of day')

plt.savefig('battery-{0}kW_c{1}-{2}-financial'.format(battery_capacity_kWh, c_rate, month))
# plt.show()



## Plot net load and NSB actions under carbon algorithms
fig = plt.figure(figsize=(13.6, 12))
gs = gridspec.GridSpec(3, 1, height_ratios=[1,1,1])

ax1 = plt.subplot(gs[0])
l3, = ax1.plot(time_axis, carbon_intensity, 
    color='r', linestyle='-', label='Carbon intensity')
ax1.set_ylabel('Carbon intensity \n(kgCO2e/kWh)')
ax1.set_xlim([time_axis[start_day*288:end_day*288][0], time_axis[start_day*288:end_day*288][-1]])
ax1.set_xticks(time_axis[start_day*288:end_day*288][::12*6])
ax1.set_xticklabels([])
ax1.axvspan(time_axis[start_day*288+int(12*7.75)],time_axis[start_day*288+12*17], 
    alpha=0.2, color='purple')
ax1.axvspan(time_axis[start_day*288+int(12*(7.75+24))],time_axis[start_day*288+12*(24+17)], 
    alpha=0.2, color='purple')
ax1.grid(False)
ax1.set_yticks([0.2,0.3,0.4,0.5,0.6,0.7])

ax3 = plt.subplot(gs[1])
ax3.axhline(y=0, linewidth=2, color='k')
l1, = ax3.plot(time_axis, battery_action_carbon,  
    linestyle='-', markersize=16, markevery=10, label='Carbon-savings')
l1, = ax3.plot(time_axis, battery_action_carbon_coopt,  
    linestyle='--', marker='o', markersize=16, markevery=10, label='Co-optimised')
ax3.set_ylabel('NSB action (kW)')
ax3.set_xticks(time_axis[start_day*288:end_day*288][::12*6])
ax3.set_xlim([time_axis[start_day*288:end_day*288][0], time_axis[start_day*288:end_day*288][-1]])
ax3.set_xticklabels([])
ax3.set_ylim([-200, 200])
ax3.set_yticks([-200,-100,0,100,200])
ax3.axvspan(time_axis[start_day*288+int(12*7.75)],time_axis[start_day*288+12*17], 
    alpha=0.2, color='purple')
ax3.axvspan(time_axis[start_day*288+int(12*(7.75+24))],time_axis[start_day*288+12*(24+17)], 
    alpha=0.2, color='purple')
ax3.legend(ncol=2, fontsize=20, loc='lower center')
ax3.grid(False)

ax4 = plt.subplot(gs[2])
ax4.axhline(y=0, linewidth=2, color='k')
l1, = ax4.plot(time_axis, net_load_no_battery,  
    linestyle='-', markersize=16, markevery=10, label='No NSB')
l1, = ax4.plot(time_axis, net_load_battery_carbon,  
    linestyle='-', markersize=16, markevery=10, label='Carbon-savings')
l1, = ax4.plot(time_axis, net_load_battery_carbon_coopt, 
    linestyle='--', marker='o', markersize=16, markevery=10, label='Co-optimised')
ax4.set_ylabel('Net load (kW)')
ax4.set_xticks(time_axis[start_day*288:end_day*288][::12*6])
ax4.set_xlim([time_axis[start_day*288:end_day*288][0], time_axis[start_day*288:end_day*288][-1]])
ax4.axvspan(time_axis[start_day*288+int(12*7.75)],time_axis[start_day*288+12*17], 
    alpha=0.2, color='purple')
ax4.axvspan(time_axis[start_day*288+int(12*(7.75+24))],time_axis[start_day*288+12*(24+17)], 
    alpha=0.2, color='purple')
ax4.legend(ncol=3, fontsize=20, loc='upper center')
ax4.grid(False)
ax4.set_yticks([-750,-500,-250,0,250,500])

ax4.set_xticks(xticks)
ax4.set_xticklabels(['0','6','12','18','0','6','13','18','0'])
ax4.set_xlabel('Time of day')

plt.savefig('battery-{0}kW_c{1}-{2}-carbon'.format(battery_capacity_kWh, c_rate, month))
# plt.show()




## Plot net load and NSB actions under self-sufficiency and ToU algorithms
fig = plt.figure(figsize=(13.6, 8))
gs = gridspec.GridSpec(2, 1, height_ratios=[1,1])

ax4 = plt.subplot(gs[0])
ax4.axhline(y=0, linewidth=2, color='k')
l1, = ax4.plot(time_axis, battery_action_sufficiency, 
    linestyle='-', markersize=16, markevery=10, label='Self-sufficiency')
l1, = ax4.plot(time_axis, ToU_battery_series,
    linestyle='--', marker='v', markersize=16, markevery=10, label='Timer')
ax4.set_ylabel('NSB action (kW)')
ax4.set_xticklabels([])
ax4.set_xlim([time_axis[start_day*288:end_day*288][0],time_axis[start_day*288:end_day*288][-1]])
ax4.set_ylim([-200, 200])
ax4.axvspan(time_axis[start_day*288+int(12*7.75)],time_axis[start_day*288+12*17], 
    alpha=0.2, color='purple')
ax4.axvspan(time_axis[start_day*288+int(12*(7.75+24))],time_axis[start_day*288+12*(24+17)], 
    alpha=0.2, color='purple')
ax4.legend(ncol=2, fontsize=20, loc='upper center')
ax4.grid(False)
ax4.set_xticks(xticks)

ax1 = plt.subplot(gs[1])
l1, = ax1.plot(time_axis, net_load_no_battery, 
    linestyle='-', markersize=16, markevery=10, label='No NSB')
l1, = ax1.plot(time_axis, net_load_battery_sufficiency,
    linestyle='-', markersize=16, markevery=10, label='Self-sufficiency')
l1, = ax1.plot(time_axis, net_load_w_ToU_battery,
    linestyle='--', marker='v', markersize=16, markevery=10, label='Timer')
ax1.set_ylabel('Net load (kW)')
ax1.set_xlim([time_axis[start_day*288:end_day*288][0],time_axis[start_day*288:end_day*288][-1]])
ax1.set_xticks(time_axis[start_day*288:end_day*288][::12*6])
ax1.set_xticklabels([])
ax1.axvspan(time_axis[start_day*288+int(12*7.75)],time_axis[start_day*288+12*17], 
    alpha=0.2, color='purple')
ax1.axvspan(time_axis[start_day*288+int(12*(7.75+24))],time_axis[start_day*288+12*(24+17)], 
    alpha=0.2, color='purple')
ax1.legend(ncol=3, fontsize=20)
ax1.grid(False)
ax1.set_xticks(xticks)
ax1.set_xticklabels(['0','6','12','18','0','6','13','18','0'])
ax1.set_xlabel('Time of day')

plt.savefig('battery-{0}kW_c{1}-{2}-self-sufficiency'.format(battery_capacity_kWh, c_rate, month))
# plt.show()


### Plot radar plot

models = ['Battery-profit', 'Communal-savings', 
 'Carbon-savings', 'Co-optimised', 'Self-sufficiency', 'Timer']

batt_revenues = np.array([battery_cost_profit, battery_cost_communal, battery_cost_sufficiency,
 battery_cost_carbon, battery_cost_carbon_coopt, battery_cost_ToU])
cust_costs = np.array([customer_cost_profit, customer_cost_communal, customer_cost_sufficiency,
 customer_cost_carbon, customer_cost_carbon_coopt, customer_cost_ToU])
sufficiencies = np.array([float(self_sufficiency_profit), float(self_sufficiency_communal), float(self_sufficiency_sufficiency),
 float(self_sufficiency_carbon), float(self_sufficiency_carbon_coopt), float(self_sufficiency_ToU)])
carbon = np.array([carbon_savings_profit, carbon_savings_communal, carbon_savings_sufficiency,
 carbon_savings_carbon, carbon_savings_carbon_coopt, carbon_savings_ToU])
transparancy = np.array([0, 1, 2, 2, 1, 9])
trust = np.array([0, 5, 5, 5, 2, 9])
simplicity = np.array([1, 1, 2, 1, 0, 4])

# Set data
df = pd.DataFrame({
'zzz': [0,0,0,0,0,0],
'Battery \nrevenue': batt_revenues/min(batt_revenues),
'Customer \nsavings': (max(cust_costs)-cust_costs)/max(max(cust_costs)-cust_costs),
'Self \nsufficiency': sufficiencies/max(sufficiencies),
'Carbon \nreductions': (carbon-min(carbon))/max(carbon-min(carbon)),
'Simplicity': simplicity/max(simplicity),
})
df.index = models
print(df)

# number of variable
categories=list(df)
N = len(categories)-1

bbox = dict(boxstyle="round", ec="white", fc="white", alpha=1)
plt.setp(ax2.get_xticklabels(), bbox=bbox)

# Initialise the spider plot 
plt.clf()
fig = plt.figure(figsize=(10,8))
ax = plt.subplot(111, polar=True)
for variation in range(6): 
    # We are going to plot the first line of the data frame.
    # But we need to repeat the first value to close the circular graph:
    values = df.iloc[variation].drop('zzz').values.flatten().tolist()
    values += values[:1]
    values
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    ax.set_yticklabels([])
    ax.plot(angles, values, linewidth=4, linestyle='solid',label=models[variation])
    # # Fill area
    ax.yaxis.grid(True,color='w',linestyle='-')
    ax.patch.set_facecolor('lightgrey')
    ax.patch.set_alpha(0.5)
    plt.xticks(angles[:-1], categories[1:])
    plt.setp(ax.get_xticklabels(), bbox=bbox)

ax.legend(ncol=1, fontsize=20, bbox_to_anchor=(1.2, 0.8))
plt.savefig('radar-{0}kW_c{1}-{2}'.format(battery_capacity_kWh, c_rate, month), bbox_inches='tight')