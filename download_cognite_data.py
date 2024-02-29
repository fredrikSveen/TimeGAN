# Imports
# %pip install seaborn
from publicdata import client as c
from functions import *
from datetime import datetime, timezone

# Nine sensors related to the inputs, outputs and controls of a compressor subsystem.
# Dictionary made with key=name of sensor and value=Id of sensor
sensors = {
    'PDT-92534':6908033636680653,
    'PT-92523':8877482139815959,
    'TIC-92504':7012228881452176,
    'TT 92532':8152209984966682,
    'FT-92537':7638223843994790,
    'TT-92539':643849686863640,
    'PT-92539':1890487216163163,
    'ZT 92543':4146236330407219,
    'KA 9101':844472910348820
}
# I'll extract a pandas dataframe for each of the sensors in sensor_names, and these will be stored
# in the following dictionary, again with the sensor name as the key.
dfs_dict = {}
full_df = pd.DataFrame()

startdate = datetime(2018, 11, 5, tzinfo=timezone.utc)
enddate = datetime(2018, 11, 15, tzinfo=timezone.utc)

# startdate = datetime(2017, 1, 1, tzinfo=timezone.utc)
# enddate = datetime(2019, 1, 1, tzinfo=timezone.utc)


#Download data from Cognite OID to a pickle file
print("Sensor name".ljust(15) + "Frame length".ljust(15) + "First timestamp".ljust(15))


# Extract dataframes from Cognite OID and store them in the dfs_dict dictionary
for k, v in sensors.items():
    res = c.time_series.data.retrieve_dataframe(id=v, 
                                                start=startdate, 
                                                end=enddate, 
                                                column_names='id',
                                                aggregates=["average"],
                                                granularity="30s",)
    # Need to find a way to grab the unit of the time-series
    res = normalize_df(res)
    res = reshape_df(res)
    dfs_dict[k] = res
    
    data_list = res[str(v)+"|average"].tolist()
    # print(data_list[:10])
    full_df[k] = data_list
    # print(f'Sensor {k} imported')
    # sr = (res.index[1] - res.index[0]).total_seconds()
    # print(f'{k}\'s shape: {res.shape} and sampling frequency: {sr} second(s)')
    print(str(k).ljust(15) + str(res.shape).ljust(15) + str(res.index[0]).ljust(15))

time_stamps = dfs_dict["PDT-92534"].index

full_df.index = time_stamps

filename = f'data/sensor_{startdate.strftime("%d_%m_%y")}to{enddate.strftime("%d_%m_%y")}'

full_df.to_csv(filename + '.csv')
full_df.to_csv(filename + '_indexless.csv', index=False)