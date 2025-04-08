#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Eva-Marie Metz
"""
import datetime
import numpy as np
import pandas as pd
import glob
import xarray as xr
import pandas as pd
from pysolar.solar import get_altitude
from zoneinfo import ZoneInfo

def CreateDF(DS, tZone):
    #print(DS.C_ppm.values)
   
    try:
        sitenamevar = DS.site_name
    except:
        sitenamevar = DS.SiteName
    #print(sitenamevar)
    try:
        Lat = DS.latitude.values
        Long = DS.longitude.values
    except:
        try:
            if sitenamevar == 'DalyRegrowth':
                Long, Lat =131.3881, -14.1592
            elif sitenamevar == 'Nimmo High Plains':
                Long, Lat = 148.5525, -36.2159
            elif sitenamevar == 'Dargo High Plains':
                Long, Lat = 147.171, -37.1334
            else:
                Lat = float(DS.latitude)
                Long = float(DS.longitude)
        except:
            Lat = np.nan
            Long = np.nan

    try:
        TimeZone = DS.time_zone
        if TimeZone == '':
            forcedfail = 1/0
    except:
        if sitenamevar in ['Fogg Dam','Howard Springs','Dry River','Daly Uncleared','Daly Pasture','Red Dirt Melon Farm','DalyRegrowth']:
            TimeZone = 'Australia/Darwin'
        elif sitenamevar in ['Dargo High Plains','Nimmo High Plains','Wallaby']:
            TimeZone = 'Australia/Sydney'
        elif sitenamevar in ['Arcturus']:
            TimeZone = 'Australia/Brisbane'
        elif tZone != 'empty': #time zone was only given in first nc file, missing in others
            TimeZone = tZone
        else:
            print('no time zone given for '+ sitenamevar)
    if TimeZone == 'NewZealand/Auckland':
        TimeZone = 'Pacific/Auckland'

    DS = DS.squeeze() # drops latitude and longitude columne which only contains one value

    try:
        d = {'Year':DS.Year.values[:],
            'Day':DS.Day.values[:], 
             'Month':DS.Month.values[:], 
             'Minute':DS.Minute.values[:],
             'Hour':DS.Hour.values[:],
             'Second':DS.Second.values[:],
             'Lat':np.ones(len(DS.Second.values[:]))*Lat,#DS.latitude.values,
             'Long':np.ones(len(DS.Second.values[:]))*Long}#DS.longitude.values}
        nanvec = np.ones(len(DS.Second.values))*np.nan
    except:
        time = DS.time.values
        years, months, days, hours, mins, secs = [],[],[],[],[],[]
        for times in time:
            years.append(int(str(times)[0:4]))
            months.append(int(str(times)[5:7]))
            days.append(int(str(times)[8:10]))
            hours.append(int(str(times)[11:13]))
            mins.append(int(str(times)[14:16]))
            secs.append(int(str(times)[17:19]))
        d = {'Year':years,
             'Day':days,
             'Month':months, 
             'Minute':mins,
             'Hour':hours,
             'Second':secs,
             'Lat':np.ones(len(DS.time.values))*Lat,#DS.latitude.values,
             'Long':np.ones(len(DS.time.values))*Long}#DS.longitude.values}    
        nanvec = np.ones(len(DS.time.values))*np.nan
    
    try:
         d.update({'Fc':DS.Fc.values[:]})
         d.update({'Fc_quality': DS.Fc_QCFlag.values[:]})
    except:
        try:
            d.update({'Fc':DS.Fco2.values[:]})
            d.update({'Fc_quality': DS.Fco2_QCFlag.values[:]})
        except:
            try:
                d.update({'Fc':DS.Fc_wpl.values[:]})
                d.update({'Fc_quality': DS.Fc_wpl_QCFlag.values[:]})
            except:
                d.update({'Fc': nanvec})
                d.update({'Fc_quality': nanvec})
    try:
         d.update({'Ts':DS.Ts.values[:]})
         d.update({'Ts_quality': DS.Ts_QCFlag.values[:]})
    except:
        d.update({'Ts': nanvec})
        d.update({'Ts_quality': nanvec})
    try:
         d.update({'Fsd_syn':DS.Fsd_syn.values[:]})
         d.update({'Fsd_syn_quality': DS.Fsd_syn_QCFlag.values[:]})
    except:
        d.update({'Fsd_syn': nanvec})
        d.update({'Fsd_syn_quality': nanvec})
    try:
        d.update({'Fsd':DS.Fsd.values[:]})
        d.update({'Fsd_quality': DS.Fsd_QCFlag.values[:]})
    except:
        d.update({'Fsd': nanvec})
        d.update({'Fsd_quality': nanvec})
    try:
         d.update({'Ta':DS.Ta.values[:]})
         d.update({'Ta_quality': DS.Ta_QCFlag.values[:]})
    except:
        d.update({'Ta': nanvec})
        d.update({'Ta_quality': nanvec})
    try:
        d.update({'CO2':DS.Cc.values[:]})
        d.update({'CO2_quality': DS.Cc_QCFlag.values[:]})
    except:
        d.update({'CO2': nanvec})
        d.update({'CO2_quality': nanvec})
    try:
         d.update({'Precipitation': DS.Precip.values[:]})
    except:
        try:
            d.update({'Precipitation': DS.Rain_W2K.values[:]})
        except:
            d.update({'Precipitation': nanvec})
    try:
         d.update({'NEE':DS.NEE.values[:]})
         d.update({'NEE_quality': DS.NEE_QCFlag.values[:]})
    except:
        d.update({'NEE': nanvec})
        d.update({'NEE_quality': nanvec})
    try:
         d.update({'NEP':DS.NEP.values[:]})
         d.update({'NEP_quality': DS.NEP_QCFlag.values[:]})
    except:
        d.update({'NEP': nanvec})
        d.update({'NEP_quality': nanvec})    
    try:
        d.update({'GPP': DS.GPP.values[:]})
        d.update({'GPP_quality': DS.GPP_QCFlag.values[:]})
    except:
        d.update({'GPP': nanvec})
        d.update({'GPP_quality': nanvec})
    try:
        d.update({'Sws': DS.Sws.values[:]})
        d.update({'Sws_quality': DS.Sws_QCFlag.values[:]})
    except:
        d.update({'Sws': nanvec})
        d.update({'Sws_quality': nanvec})
    try:
        d.update({'ER_dark': DS.ER_dark.values[:]})
        d.update({'ER_dark_quality': DS.ER_dark_QCFlag.values[:]})
    except:
        d.update({'ER_dark': nanvec})
        d.update({'ER_dark_quality': nanvec})
    try:
        d.update({'ER_night': DS.ER_night.values[:]})
        d.update({'ER_night_quality': DS.ER_night_QCFlag.values[:]})
    except:
        d.update({'ER_night': nanvec})
        d.update({'ER_night_quality': nanvec})
    df = pd.DataFrame(data=d)

    return df, TimeZone

def CreateDataFrameOzFlux(station_id,level):
    folderPath =savepath
    if level == 'L6':
        folderPath =savepath +  "L6/"
    print("Reading data:" +station_id)
    filepath1 = folderPath+station_id+"*.nc"#_L6_CPartitioned.nc"
    tZone = 'empty'
    for num, filepath in enumerate(glob.glob(filepath1)):    
        print(filepath)
        try:
            DS = xr.open_dataset(filepath)
        except:
            DS = xr.open_dataset(filepath, drop_variables = 'time')
        if num == 0:        
        #create Dataframe
            df, tZone = CreateDF(DS,tZone)
        else:      
        #create Dataframe
            df3, tZone= CreateDF(DS,tZone)
            df = df.append(df3, ignore_index=True)

    if station_id in ['Samford']: #from mgCO2/m2/s -> umol/m^2/s
        fcnew = df.apply(lambda x: x.Fc if x.Year >= 2021 else x.Fc *1000/44, axis =1)
        df.drop(columns='Fc', inplace = True)
        df.insert(0, column = 'Fc', value = fcnew)
        del fcnew

    df.to_pickle(savepath + "/DF15_noDate_"+level+station_id+".pkl")
    #create date variable
    print("create timestamp")
    datetimearray = df.apply(lambda x: datetime.datetime(int(x.Year),int(x.Month),int(x.Day),int(x.Hour),int(x.Minute),int(x.Second),tzinfo=ZoneInfo(tZone)).astimezone(ZoneInfo('UTC')),axis=1)
       
    df.insert(loc=1,column='DateTime',value=datetimearray)
    df.rename(columns = {'Year':'YearLocalT','Day':'DayLocalT', 'Month':'MonthLocalT', 'Minute':'MinuteLocalT','Hour':'HourLocalT','Second':'SecondLocalT'}, inplace = True)
    date = df.apply(lambda x: datetime.date(int(x.DateTime.year),int(x.DateTime.month),int(x.DateTime.day)),axis=1)
    df.insert(loc=1,column='Date',value=date)

    try: #depends on pandas version
        tup = df.apply(lambda x: (int(x.Date.year),int(x.Date.month),int(x.Date.day)),axis=1)
    except:
        tup = df.apply(lambda x: (int(x.Date[0:4]),int(x.Date[5:7]),int(x.Date[8:10])),axis=1)
    df.insert(0,column = 'Year', value= [i[0] for i in tup])
    df.insert(0,column = 'Month', value= [i[1] for i in tup])
    df.insert(0,column = 'Day', value= [i[2] for i in tup])

    #add day lentgh inforamtion
    altitude = df.apply(lambda x: get_altitude(x.Lat,x.Long,datetime.datetime(int(x.DateTime.year),int(x.DateTime.month),int(x.DateTime.day),int(x.DateTime.hour),int(x.DateTime.minute),int(x.DateTime.second),0,tzinfo=datetime.timezone.utc)),axis=1)
    df.insert(loc=1,column='sun_altitude',value=altitude)

    print('save data')
    #from version 10 local time considered
    df.to_pickle(savepath + "/DF15_"+level+station_id+".pkl")
    df.to_csv(savepath + "/DF15_"+level+station_id+".csv")

#main
savepath = ''#path to OzFlux data    
filenamelist = ['AdelaideRiver','AliceSpringsMulga',
                'Boyagin1','Calperum','CapeTribulation',
                'Collie', 'CowBay','CumberlandMelaleuca','CumberlandPlain',
                'DalyPasture','DalyRegrowth','Dargo','DryRiver',
                'Emerald','fallscreek','FoggDam','GatumPasture','Gingin','GWW','HowardSprings',
                'HowardUnderstory','Litchfield','Longreach','Nimmo',
                'Otway','RDMF','Ridgefield',
                'Riggs','Robson','Samford',
                'SturtPlains','TiTreeEast','Tumbarumba','Wallaby','Warra',
                'Whroo','WombatStateForest','Yanco','YarCon','YarIrr']


for name in filenamelist:
    CreateDataFrameOzFlux(name,'') #'L6' for L6, '' for L3
