#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 11:50:45 2025

"""


import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import scipy.stats 
import xarray as xr
import geopandas
from matplotlib.patches import Rectangle
import os
from scipy.stats import linregress
import datetime


#### README
# With this Script Fig. 1-5 and A1 -A11, except for Fig. 3 can be created 
# Please create the LPJ and OZFlux Dataframes first by running Code_Metz2025_LpjDFs.py and Code_Metz2025_OzFluxDFs.py
# Please adapt line 480 to obtain all sensitivity plots such as Figure 2, Please adapt FigureA5Panel below to get one panel of Figure A5 with each run
# Please run Code_Metz2025_Plot_Carpet.py to obtain Figure 3

FigureA5Panel = 0 # integer from 0 to 7 for panels of Figure A5

savepath = ""

##### FUNCTIONS #####

def ReadAndQFilterOzFluxDaily(StationName):
    #read prepared dataframes
    if os.path.isfile(savepath+"/DF18_qfiltered_"+StationName+".pkl"):
        dfStation = pd.read_pickle(savepath+"/DF18_qfiltered_"+StationName+".pkl")
        dfStation_night = pd.read_pickle(savepath+"/DF18_qfiltered_night_"+StationName+".pkl")
        if len(dfStation.Lat) > 0:
            lats, longs = dfStation.Lat.values[0], dfStation.Long.values[0]
        else:
            lats, longs = np.nan, np.nan
    else:
        #getOzflux
        dfStation = pd.read_csv(savepath+"/DF15_"+StationName+".csv")

        #select according to quality flag, 0 is the only valid value
        #https://github.com/OzFlux/PyFluxPro/wiki/QC-Flag-Definitions
        dfStation.loc[dfStation[(dfStation['Fc_quality'] > 0)].index,
                                    'Fc'] = np.nan
        dfStation.loc[dfStation[(dfStation['Ts_quality'] > 0)].index,
                                    'Ts'] = np.nan
        dfStation.loc[dfStation[(dfStation['Ta_quality'] > 0)].index,
                                    'Ta'] = np.nan
        dfStation.loc[dfStation[(dfStation['Sws_quality'] > 0)].index,
                                    'Sws'] = np.nan
        # set unreasonable values to nan
        dfStation.loc[dfStation[(dfStation['Fc'] <= -9999)].index,
                                    'Fc'] = np.nan
        dfStation.loc[dfStation[(dfStation['Ts'] <= -9999)].index,
                                    'Ts'] = np.nan
        dfStation.loc[dfStation[(dfStation['Ta'] <= -9999)].index,
                                    'Ta'] = np.nan
        if StationName == 'TiTreeEast':#TTE has 1,5 years of SWS ==0, should be excluded
            dfStation.loc[dfStation[(dfStation['Sws'] <= 0)].index,
                                    'Sws'] = np.nan 
        else:
            dfStation.loc[dfStation[(dfStation['Sws'] < 0)].index,
                                    'Sws'] = np.nan 
        dfStation.loc[dfStation[(dfStation['Precipitation'] < 0)].index,
                                    'Precipitation'] = np.nan

        #drop all half hour values where at least on of the three variabiles Fc, Sws, Ts has no measurement
        dfStation.dropna(0,subset=['Fc','Sws','Ts'], inplace = True)

        lats = dfStation.Lat.values[0]
        longs = dfStation.Long.values[0]
        if np.isnan(lats):
            print('no data for this dataset')
            return np.nan, np.nan, lats, longs

        #add date parameters
        if 'Year' not in dfStation.keys(): 
            tup = dfStation.apply(lambda x: (int(x.Date[0:4]),int(x.Date[5:7]),int(x.Date[8:10])),axis=1)
            dfStation.insert(0,column = 'Year', value= [i[0] for i in tup])
            dfStation.insert(0,column = 'Month', value= [i[1] for i in tup])
            dfStation.insert(0,column = 'Day', value= [i[2] for i in tup])
        
        dates = dfStation.apply(lambda x: datetime.date(x.Year,x.Month,x.Day),axis = 1)
        dfStation.drop(columns='Date', inplace = True)
        dfStation.insert(0,column = 'Date', value = dates)

        #select for night measurements
        dfStation_night = dfStation[(dfStation.sun_altitude < 0)]   #calculated with pysolar.solar

        #save half hourly data as datframe
        dfStation.to_pickle(savepath+"/DF18_qfiltered_HalfHourly_"+StationName+".pkl")
        dfStation_night.to_pickle(savepath+"/DF18_qfiltered_HalfHourly_night_"+StationName+".pkl")
            
        #calculate nighttime means
        dfStation_nightNum = dfStation_night.groupby(['Date','Year','Month','Day'])['Fc'].count().reset_index()
        dfStation_nightNum.rename(columns={'Fc':'num'},inplace=True)
        dfStation_night = dfStation_night.groupby(['Date','Year','Month','Day'])['Fc','Sws','Ts','Ta','Precipitation','Lat', 'Long'].mean().reset_index()
        
        #drop days with less than 5 measurements
        dfStation_night = pd.merge(dfStation_night,dfStation_nightNum, on =['Date','Year','Month','Day'] , how = 'outer')
        dfStation_night.loc[dfStation_night[(dfStation_night['num'] < 5)].index,
                                    'Fc'] = np.nan
        dfStation_night.dropna(0,subset=['Fc','Sws'], inplace = True)
        #dfStation_night.drop(columns= 'num', inplace = True)
        
        #calculate day means
        dfStationNum = dfStation.groupby(['Date','Year','Month','Day'])['Fc'].count().reset_index()
        dfStationNum.rename(columns={'Fc':'num'},inplace=True)
        
        dfStation = dfStation.groupby(['Date','Year','Month','Day'])['Fc','Sws','Ts','Ta','Precipitation','Lat', 'Long'].mean().reset_index()
        #drop days with less than 5 measurements
        dfStation = pd.merge(dfStation,dfStationNum, on =['Date','Year','Month','Day'] , how = 'outer')
        dfStation.loc[dfStation[(dfStation['num'] < 5)].index,
                                    'Fc'] = np.nan
        dfStation.dropna(0 ,subset=['Fc','Sws'],inplace = True)
        #dfStation.drop(columns= 'num', inplace = True)

        print('saving')
        dfStation.insert(0,column = 'DailyPrecipitation', value= dfStation.Precipitation*24*2) #from 30 min mean to daily
        dfStation.to_pickle(savepath+"/DF18_qfiltered_"+StationName+".pkl")
        dfStation_night.to_pickle(savepath+"/DF18_qfiltered_night_"+StationName+".pkl")
    
    dfStation = dfStation[(dfStation.Year < 2024)]
    dfStation_night = dfStation_night[(dfStation_night.Year < 2024)]

    return dfStation, dfStation_night, lats, longs

def getNumDayOfMonth(year,month):
    """returns list of number of days within given month in given year"""
    listDays = [31,28,31,30,31,30,31,31,30,31,30,31]
    listDaysl = [31,29,31,30,31,30,31,31,30,31,30,31]
    if year < 1900:
        print('year is out of implemented range, check code')
    elif year in list(range(1904,2100,4)):
        days = listDaysl[month-1]
    else:
        days = listDays[month-1]

    return days

def getReferenceDateDay(year_min,year_max,month_min,month_max):
    yea = []
    mon = []
    day = []
    numday = []
    for k in range(year_min,year_max +1):
        len0 = len(yea)
        if k == year_min:
            for p in range(month_min,12+1):
                for d in range(1,getNumDayOfMonth(k,p)+1):
                    yea.append(k)
                    mon.append(p)
                    day.append(d)
        elif k == year_max:
            for p in range(1,month_max+1):
                for d in range(1,getNumDayOfMonth(k,p)+1):
                    yea.append(k)
                    mon.append(p)
                    day.append(d)
        else:
            for p in range(1,13):
                for d in range(1,getNumDayOfMonth(k,p)+1):
                    yea.append(k)
                    mon.append(p)
                    day.append(d)
        numday = numday + list(range(len(yea)-len0))
        

    dateData = {"Year":yea,"Month":mon,"Day":day,"NumDayInYear":numday}
    DateRef = pd.DataFrame(data=dateData)
    datesDR = DateRef.apply(lambda x: np.datetime64(str(x.Year)+'-'+str(x.Month).zfill(2)+'-'+str(x.Day).zfill(2)),axis = 1)
    DateRef.insert(0,column='Date',value=datesDR)
    return DateRef



##### SETTINGS #####

minMeasnum = 100 # minimum number of daily values to calculate the sensitivities

wetstations = ['Ridgefield', 'CumberlandPlain','GWW', 'Yanco','YarCon', 'GatumPasture', 'Whroo', 'YarIrr', 'Riggs', 'AdelaideRiver', 'WombatStateForest', 'CapeTribulation', 'Nimmo', 'Otway', 'Tumbarumba', 'Longreach', 'Robson', 'CowBay', 'SturtPlains', 'Warra', 'Wallaby', 'Samford', 'Dargo', 'fallscreek', 'FoggDam']
drystations = ['TiTreeEast', 'Calperum', 'Gingin', 'Boyagin1', 'AliceSpringsMulga', 'RDMF', 'DalyUncleared', 'Collie', 'DryRiver', 'Litchfield', 'DalyPasture', 'CumberlandMelaleuca', 'Emerald', 'HowardSprings', 'HowardUnderstory']

stationlist = ['AdelaideRiver','AliceSpringsMulga',
                'Boyagin1','Calperum','CapeTribulation',
                'Collie', 'CowBay','CumberlandMelaleuca','CumberlandPlain',
                'DalyPasture','DalyUncleared','Dargo','DryRiver',
                'Emerald','fallscreek','FoggDam','GatumPasture','Gingin','GWW','HowardSprings',
                'HowardUnderstory','Litchfield','Longreach','Nimmo',
                'Otway','RDMF','Ridgefield',
                'Riggs','Robson','Samford',
                'SturtPlains','TiTreeEast','Tumbarumba','Wallaby','Warra',
                'Whroo','WombatStateForest','Yanco','YarCon','YarIrr']    
stationlist_fullnames= ['Adelaide River', 'Alice Springs Mulga',
                'Boyagin', 'Calperum','Cape Tribulation',
                'Collie', 'Cow Bay','Cumberland Melaleuca','Cumberland Plain',
                'Daly River Pasture','Daly River Uncleared','Dargo','Dry River',
                'Arcturus','Alpine Peatland','Fogg Dam','Gatum Pasture','Gingin','Great Western Woodlands','Howard Springs', 
                'Howard Springs Understory','Litchfield','Mitchell Grass Rangeland','Nimmo',
                'Otway', 'Red Dirt Melon Farm','Ridgefield', 
                'Riggs Creek','Robson Creek','Samford',
                'Sturt Plains','Ti Tree East','Tumbarumba','Wallaby Creek','Warra',
                'Whroo','Wombat','Yanco','Yarramundi Control', 'Yarramundi Irrigated']
stationlistLPJ =  ['AdelaideRiver','ASM',
                'Boyagin','Calperum','CapeTribulation',
                'Collie','CowBay','Cumberland_YarCon_YarIrr','Cumberland_YarCon_YarIrr',
                'DalyUncleared_DalyRegrowth','DalyUncleared_DalyRegrowth','Dargo','DryRiver',
                'Emerald','fallscreek','FoggDam','GatumPasture','Gingin','GWW','Howard',
                'Howard','Litchfield','Longreach','Nimmo',
                'Otway','RDMF','Ridgefield',
                'Riggs','Robson','Samford',
                'DalyPasture_SturtPlains','TTE','Tumbarumba','Wallaby','Warra',
                'Whroo','WombatStateForest','Yanco','Cumberland_YarCon_YarIrr','Cumberland_YarCon_YarIrr']
regs = ['N','N','SW','SE','NE',
        'SW','NE','SE','SE',
        'N','N','SE','N',
        'NE','SE','N','SE','SW','SW','N',
        'N','N','NE','SE',
        'SE','N','SW',
        'SE','NE','NE',
        'N','N','SE','SE','SE',
        'SE','SE','SE','SE','SE']
lats = [-13.0769, -22.2830, -32.477093, -34.0027, -16.103219, -33.42, -16.238189, -33.613996, -33.615278, -14.0633, -14.1592, -37.1334, -15.2588, -23.85872, -36.862222, -12.5452, -37.390033, -31.37635, -30.19140, -12.49520, -12.49520, -13.179041780, -23.523265764497292, -36.21590, -38.52450, -14.563639, -32.5061020, -36.656, -17.11746943, -27.38806, -17.1507, -22.283, -35.65660, -37.42590, -43.09502, -36.6732, -37.42220, -34.98780, -33.620812, -33.620812]
longs = [131.1178, 133.2490, 116.938559, 140.5875, 145.446922, 116.237, 145.42715, 150.726418, 150.723611, 131.3181, 131.3881, 147.171, 132.3706, 148.4746, 147.320833, 131.3072, 141.960897, 115.71377, 120.654167, 131.15005, 131.15005, 130.7945459, 144.31041571964494, 148.55250, 142.810, 132.477567, 116.9668270, 145.576, 145.6301375, 152.877780, 133.3502, 133.2490, 148.1516, 145.18780, 146.65452, 145.0294, 144.09440, 146.29090, 150.7632840, 150.763284]

LPJ_db_str = '_AllOzFlux_default_wTs'
FigureA5 = Figure2 = False

#Infodataset
StationParameters = pd.DataFrame(data={'Fullname':stationlist_fullnames,
                                        'lat':lats,
                                        'long':longs,
                                        'regs':regs,
                                        'station':stationlist})
# add aridity information to dataset
dsAI_AU = xr.open_dataset(savepath+"/ai_v3_yr_AU.nc")
AIl = []
for i in range(len(StationParameters.lat)):
    AIl.append(float(dsAI_AU.interp(latitude = StationParameters.lat.values[i],longitude=StationParameters.long.values[i], method = 'nearest').AI.values))        
StationParameters.insert(0, column = 'AI',value = AIl)    

litterlist = []
for stid in range(len(StationParameters.station)):
    #getOzflux
    dfStation0, dfStation_night0, lats0, longs0 = ReadAndQFilterOzFluxDaily(StationParameters.station.values[stid])
    lpjname = stationlistLPJ[int(np.where(np.array(stationlist) == StationParameters.station.values[stid])[0][0])]
    #get LPJ if sort by litter
    dfStationLPJ = pd.read_pickle(savepath+"/"+lpjname+LPJ_db_str+"_Annual_v0.pkl")
    #take only years also present in OuFlux
    dfStationLPJ = dfStationLPJ[(dfStationLPJ.Year <= dfStation0.Year.max())&(dfStationLPJ.Year >= dfStation0.Year.min())]
    litterlist.append(dfStationLPJ.litter.mean())
    
StationParameters.insert(0,column='Litter_new',value=litterlist)
StationParameters.to_pickle(savepath + '/StationParam.pkl')
StationParameters.to_csv(savepath + '/StationParam.csv')

##### MAIN ######

First = True
FirstA = True
firstT = True
first = True
firstA5 = True
timelim = ''

#sort stations by mean sws
# get mean sws list
sws_dic = {}
for stid in range(len(stationlist)):
    
    #getOzflux
    dfStation, dfStation_night, lats, longs = ReadAndQFilterOzFluxDaily(stationlist[stid])
    
    if stationlist[stid] in ['Boyagin_understory','Digby','DalyRegrowth']:
        sws_dic[stationlist[stid]] = 0
    else:
        sws_dic[stationlist[stid]] = dfStation_night.Sws.mean()
sws_dic_sort = dict(sorted(sws_dic.items(), key=lambda item: item[1]))

# Senitivity Figures: Figure 2, 5, S5, S8, S9
# Heatmaps and scatter S2, S4, S6
# loop over stations
for stid in range(len(stationlist)):
    
    #sort by OzFlux soil moisture
    stationname = list(sws_dic_sort.keys())[stid]#stationlist[stid]
    stationnamelist = list(sws_dic_sort.keys())
    stationnameLPJ = stationlistLPJ[stationlist.index(stationname)]
    stationfullname = stationlist_fullnames[stationlist.index(stationname)]
    
    #getOzflux
    dfStation, dfStation_night, lats, longs = ReadAndQFilterOzFluxDaily(stationname)
    #2do remove
    if len(dfStation) == 0:
        print('no data for '+stationname)
        continue
    
    #read LPJ grid cell run
    dfStationLPJ = pd.read_pickle(savepath+"/"+stationnameLPJ+LPJ_db_str+"_v0.pkl")
    #take only years also present in OuFlux
    dfStationLPJ = dfStationLPJ[(dfStationLPJ.Year <= dfStation.Year.max())&(dfStationLPJ.Year >= dfStation.Year.min())]
    
    dfStationLPJA = pd.read_pickle(savepath+"/"+stationnameLPJ+LPJ_db_str+"_Annual_v0.pkl")
    dfStationLPJA = dfStationLPJA[(dfStationLPJA.Year <= dfStation.Year.max())&(dfStationLPJA.Year >= dfStation.Year.min())]
    
    try:
        dfStation.drop(columns = 'Date',inplace = True)
    except:
        pass
    try:
        dfStation_night.drop(columns = 'Date',inplace = True)
    except:
        pass

    dfStationmerge = pd.merge(dfStation, dfStationLPJ, on=['Year', 'Month', 'Day'], how = 'inner')
    dfStationmerge_night = pd.merge(dfStation_night, dfStationLPJ, on=['Year', 'Month', 'Day'], how = 'inner')
    dfStationmerge.insert(0, column = 'ra', value = dfStationmerge.gpp-dfStationmerge.npp)   
    dfStationmerge_night.insert(0, column = 'ra', value = dfStationmerge_night.gpp-dfStationmerge_night.npp)   
    dfStationmerge_night = pd.merge(dfStationmerge_night, dfStation[['Year', 'Month', 'Day','DailyPrecipitation']], on=['Year', 'Month', 'Day'], how = 'left')
    


    # Figure A2 Scatter TER Soil moisture for indiv Temp classes
    if True:##
        if stationname == 'AliceSpringsMulga':
            fig, ax2 = plt.subplots()
            ax2.plot(dfStationmerge_night.Sws, dfStationmerge_night.Fc, ls = '', marker = '.')
            if len(dfStationmerge_night.Sws) > 0:
                slope, intercept, correl, p, perr = linregress(dfStationmerge_night.Sws, dfStationmerge_night.Fc)
            
                ax2.set_ylabel('OzFlux TER '+ r'$\rm [\mu mol/m^2/s]$')
                ax2.set_xlabel('OzFlux Soil Moisture [%]')
                ax2.set_ylim(dfStationmerge_night.Fc.quantile(0.005),dfStationmerge_night.Fc.quantile(0.995))
                ax2.set_xlim(dfStationmerge_night.Sws.min(),dfStationmerge_night.Sws.quantile(0.999))
                
                T_board = [[0,5],[5,10],[10,15],[15,20],[20,25],[25,30],[30,35],[35,40],[40,45]]
                T_board_color = ['#440154FF','#404788FF','#2D708EFF','#1F968BFF','#55C667FF','#95D840FF','#DCE319FF','#FDE725FF','orange']
            
                for Ti2 in range(len(T_board)):
                    try:
                        dfStationmerge_nightT = dfStationmerge_night[(dfStationmerge_night.Ts <= T_board[Ti2][1])&(dfStationmerge_night.Ts > T_board[Ti2][0])]
                        if len(dfStationmerge_nightT) >= 100:
                            ax2.plot(dfStationmerge_nightT.Sws, 
                                        dfStationmerge_nightT.Fc, 
                                        ls = '', marker = '.',color = T_board_color[Ti2])
                            slopeS, interceptS, correlS, pS, perrS = linregress(dfStationmerge_nightT.Sws, dfStationmerge_nightT.Fc)
                    
                            ax2.plot([0,dfStationmerge_night.Sws.quantile(0.999)],
                                    [interceptS + slopeS*0,interceptS + slopeS*dfStationmerge_night.Sws.quantile(0.999)],
                                    ls = ':',color = T_board_color[Ti2], label = 'T=['+str(T_board[Ti2][0])+','+str(T_board[Ti2][1])+'] r²='+str(round(correlS**2,2)) + ', slope='+str(round(slopeS,1))+'+-'+str(round(perrS,1)))   
                    except:
                        continue
            else:
                ax2.text(0,0,'no data')
            ax2.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            plt.title(stationlist_fullnames[np.where(np.array(stationlist) == stationname)[0][0]])
            plt.savefig(savepath+'/FigureA2.png',bbox_inches='tight',dpi=300)
            plt.savefig(savepath+'/FigureA2.eps',bbox_inches='tight',dpi=300)
            plt.savefig(savepath+'/FigureA2.pdf',bbox_inches='tight',dpi=300,format = 'pdf')

    #Figure A7     
    if True:# Timeseries SWS and Rh 2 panels
        if stationname in ['Gingin','AliceSpringsMulga']:
            year = 2014
            RefDate = getReferenceDateDay(year,year,1,12)
            dfStationmerge_night_ts = dfStationmerge_night[(dfStationmerge_night.Year == year)]
            
            dfStationmerge_night_ts_nan = pd.merge(RefDate,dfStationmerge_night_ts,how='left', on = 'Date')
            fig, ax2 = plt.subplots(2,1)
            conv_fact = 10**6/(12*24*60*60)#gC /m^2 / day -> mu mol/m^2 /s
            ax2[0].plot(dfStationmerge_night_ts_nan.Date, dfStationmerge_night_ts_nan.swc1, color = 'black',ls = '-',label = 'LPJ')
            ax2[0].plot(dfStationmerge_night_ts_nan.Date, dfStationmerge_night_ts_nan.Sws, color = 'blue',ls = '-',label = 'OzFlux')
            ax2[1].plot(dfStationmerge_night_ts_nan.Date, dfStationmerge_night_ts_nan.Fc, color = 'green',ls = '-',label= 'OzFlux Nighttime NEE')
            ax2[1].plot(dfStationmerge_night_ts_nan.Date, (dfStationmerge_night_ts_nan.ra+dfStationmerge_night_ts_nan.rh)*conv_fact, color = 'black',ls = '-',label= 'LPJ Ra+Rh')
            ax2[0].plot(dfStationmerge_night_ts.Date, dfStationmerge_night_ts.swc1, color = 'black',ls = ':')#,label = 'LPJ')
            ax2[0].plot(dfStationmerge_night_ts.Date, dfStationmerge_night_ts.Sws, color = 'blue',ls = ':')#,label = 'OzFlux')
            ax2[1].plot(dfStationmerge_night_ts.Date, dfStationmerge_night_ts.Fc, color = 'green',ls = ':')#,label= 'OzFlux Nighttime NEE')
            ax2[1].plot(dfStationmerge_night_ts.Date, (dfStationmerge_night_ts.ra+dfStationmerge_night_ts.rh)*conv_fact, color = 'black',ls = ':')#,label= 'LPJ Ra+Rh')
            
            ax2[0].legend()
            ax2[1].legend(ncol=2)
            ax2[0].set_ylabel('Soil Moisture')
            ax2[1].set_ylabel('Resp. [mu mol/(m2 s)]')
            ax2[1].set_xlabel('Date')
            plt.savefig(savepath+'/FigureA7_'+stationname+'_'+str(year)+'.png',bbox_inches='tight',dpi=300)
    
    
    #Figure A4 and A6 Heat map 5deg T intervals
    if True: 
        usedplots = {'DryRiver':[35,40],'DalyUncleared':[30,35],'HowardSprings':[25,30],'YarCon':[15,20],'Gingin':[15,20],'Robson':[15,20]}
        if stationname in usedplots.keys():
            if len(dfStationmerge_night.Sws) > 0:
                T_board = usedplots[stationname]
                Tvar = 'Ts'#'TsoilOrig''tair'#'Ts'
                dfStationmerge_nightT = dfStationmerge_night[(dfStationmerge_night[Tvar] <= T_board[1])&(dfStationmerge_night[Tvar] > T_board[0])]
                if len(dfStationmerge_nightT.Sws) > 0: 
                         
                    #OzFlux
                    numBins = 30
                    d_diff = (dfStationmerge_nightT.Fc.max()-dfStationmerge_nightT.Fc.min())/numBins # 30bins
                    d_para = (dfStationmerge_nightT.Sws.max()-dfStationmerge_nightT.Sws.min())/numBins
                    
                    xedges = np.array(range(numBins))*d_para + dfStationmerge_nightT.Sws.min()
                    yedges = np.array(range(numBins))*d_diff + dfStationmerge_nightT.Fc.min()
                    
                    H, xedges, yedges = np.histogram2d(dfStationmerge_nightT.Sws,dfStationmerge_nightT.Fc, bins=(xedges, yedges))

                    H2 = H.T
                    H2normed = H2 / H2.max(axis=0)
                    X, Y = np.meshgrid(xedges, yedges)

                    fig, ax = plt.subplots()
                    pc = ax.pcolormesh(X, Y, H2, cmap = 'gnuplot')
                    ax.set_xlabel('Sws')
                    ax.set_ylabel('TER'+ r'$\rm [\mu mol/m^2/s]$')
                    fig.colorbar(pc)
                    plt.title(stationname + ' '+str(T_board[0])+ '°C-'+str(T_board[1])+ '°C')
                    plt.savefig(savepath+'/HeatmapOzFlux_'+stationname+str(T_board[1])+'_HeatScatter_OzFlux.png',bbox_inches='tight',dpi=200)
                    
                    if stationname in ['Gingin','Robson']: #LPJ TER
                        dfStationmerge_nightT = dfStationmerge_night[(dfStationmerge_night['tair'] <= T_board[1])&(dfStationmerge_night['tair'] > T_board[0])]
                        conv_fact = 10**6/(12*24*60*60)#gC /m^2 / day -> mu mol/m^2 /s
                        numBins = 30
                        d_diff = conv_fact*((dfStationmerge_nightT.ra+dfStationmerge_nightT.rh).quantile(0.99)-(dfStationmerge_nightT.ra+dfStationmerge_nightT.rh).min())/numBins # 30bins
                        d_para = (dfStationmerge_nightT.swc1.max()-dfStationmerge_nightT.swc1.min())/numBins
                        
                        xedges = np.array(range(numBins))*d_para + dfStationmerge_nightT.swc1.min()
                        yedges = np.array(range(numBins))*d_diff + conv_fact*(dfStationmerge_nightT.ra+dfStationmerge_nightT.rh).min()
                        
                        H, xedges, yedges = np.histogram2d(dfStationmerge_nightT.swc1,conv_fact*(dfStationmerge_nightT.ra+dfStationmerge_nightT.rh), bins=(xedges, yedges))
                    
                        H2 = H.T
                        H2normed = H2 / H2.max(axis=0)
                        X, Y = np.meshgrid(xedges, yedges)

                        fig2, ax2 = plt.subplots()
                        pc2 = ax2.pcolormesh(X, Y, H2, cmap = 'gnuplot')
                        ax2.set_xlabel('Sws')
                        ax2.set_ylabel('TER'+ r'$\rm [\mu mol/m^2/s]$')
                        fig2.colorbar(pc2)
                        plt.title(stationname + ' '+str(T_board[0])+ '°C-'+str(T_board[1])+ '°C')
                        plt.savefig(savepath+'/HeatmapLPJ_'+stationname+str(T_board[1])+'_HeatScatter_OzFlux.png',bbox_inches='tight',dpi=300)
                    
        
    # Figure A5 sensitivities individually for each T interval
    if True:    #sensitivities OzFlux for different Temps
        FigureA5 = True
        if firstA5:
            firstA5 = False
            fig4, ax4 = plt.subplots(figsize=(6,3))
            BGcorrs = []

        T_board = [[0,5],[5,10],[10,15],[15,20],[20,25],[25,30],[30,35],[35,40],[40,45]]
        textlist= ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)']
        anovalist=['nan',0.1959,0.1763,'0.1090',0.0046,0.0001,'0.00002',0.0685] # taken from ANOVA results
            
        for TiA5 in [FigureA5Panel]:# insert value from 0 - 7 to get different temp intervals
            try:
                #only calculate and plot if N >= minMeasNumber
                if len(dfStationmerge_night[(dfStationmerge_night.Ts <= T_board[TiA5][1])&(dfStationmerge_night.Ts > T_board[TiA5][0])].Sws) >= minMeasnum: 
                    
                    slope, intercept, correl, p, perr = linregress(dfStationmerge_night[(dfStationmerge_night.Ts <= T_board[TiA5][1])&(dfStationmerge_night.Ts > T_board[TiA5][0])].Sws, dfStationmerge_night[(dfStationmerge_night.Ts <= T_board[TiA5][1])&(dfStationmerge_night.Ts > T_board[TiA5][0])].Fc)
                    #for single color plot
                    ax4.plot(stid +0.5, slope, marker = 'x', color = 'black', markersize = 4)
                    ax4.errorbar(stid +0.5, slope, perr, color = 'black')   
                    BGcorrs.append(correl**2)     
                else:
                    BGcorrs.append(np.nan)      
            except:
                BGcorrs.append(np.nan) 
                    
      
    #THESE SETTINGS MUST BE VARIED TO OBTAIN ALL PLOTS!
    plotdataset = 'OzFlux'#'OzFlux' or 'LPJ'
    tempvar = 'TsoilOrig' #which Temperature to use for LPJ 'TsoilOrig' or 'tair'
    LPJParam = 'rh' # which resp variable of LPJ to use 'TER','rh','ra'

    if True:   
        Figure2 = True 
        if first:
            #parameter lists for saving output
            corrs = []
            Stationcorrlist = []
            Stationfullnamecorrlist = []
            templist = []
            corr_r = []
            slopesl = []
            slopesLPJlTER = []
            slopesLPJlrh = []
            slopesLPJlra = []
            numberMeas = []
            Swslist = []
            Fclist =[]
            first = False
            
            fig5, ax5 = plt.subplots()              

        T_board = [[0,5],[5,10],[10,15],[15,20],[20,25],[25,30],[30,35],[35,40],[40,45]]
        T_board_color = ['#440154FF','#404788FF','#2D708EFF','#1F968BFF','#55C667FF','#95D840FF','#DCE319FF','#FDE725FF','orange']

        Stationcorrlist.append(stationname)
        templist.append(9999)
        Stationfullnamecorrlist.append(stationfullname)
        MeanSws = dfStationmerge_night.Sws.mean()*100
        Swslist.append(dfStationmerge_night.Sws.mean())
        Fclist.append(dfStationmerge_night.Fc.mean())
        numberMeas.append(len(dfStationmerge_night.Sws))
        slopesl.append(np.nan)
        corrs.append(np.nan)
        corr_r.append(np.nan)
        slopesLPJlTER.append(np.nan)
        slopesLPJlrh.append(np.nan)
        slopesLPJlra.append(np.nan)

        for Ti in range(len(T_board)-1):
            Stationcorrlist.append(stationname)
            templist.append(T_board[Ti][0])
            Stationfullnamecorrlist.append(stationfullname)
           
            try:
                #only calculate and plot if N >= minMeasnum
                if len(dfStationmerge_night[(dfStationmerge_night.Ts <= T_board[Ti][1])&(dfStationmerge_night.Ts > T_board[Ti][0])].Sws) >= minMeasnum: 
                    MeanSwsT = dfStationmerge_night[(dfStationmerge_night.Ts <= T_board[Ti][1])&(dfStationmerge_night.Ts > T_board[Ti][0])].Sws.mean()
                    Swslist.append(MeanSwsT)
                    Fclist.append(dfStationmerge_night[(dfStationmerge_night.Ts <= T_board[Ti][1])&(dfStationmerge_night.Ts > T_board[Ti][0])].Fc.mean())

                    slope, intercept, correl, p, perr = linregress(dfStationmerge_night[(dfStationmerge_night.Ts <= T_board[Ti][1])&(dfStationmerge_night.Ts > T_board[Ti][0])].Sws, dfStationmerge_night[(dfStationmerge_night.Ts <= T_board[Ti][1])&(dfStationmerge_night.Ts > T_board[Ti][0])].Fc)
                    if plotdataset == 'OzFlux':
                        savestr = plotdataset
                        ax5.plot(stid +0.5, slope, marker = 'x', color = T_board_color[Ti], markersize = 4)
                        ax5.errorbar(stid +0.5, slope, perr, color = T_board_color[Ti])
                        #ax5.plot(MeanSws, slope, marker = 'x', color = T_board_color[Ti], markersize = 4)
                        #ax5.errorbar(MeanSws, slope, perr, color = T_board_color[Ti])
                    
                    slopesl.append(slope)
                    corrs.append(correl**2)
                    corr_r.append(correl)
                    
                else:
                    slopesl.append(np.nan)
                    corrs.append(np.nan)
                    corr_r.append(np.nan)
                    Swslist.append(np.nan)
                    Fclist.append(np.nan)
                numberMeas.append(len(dfStationmerge_night[(dfStationmerge_night.Ts <= T_board[Ti][1])&(dfStationmerge_night.Ts > T_board[Ti][0])].Sws))

            except:
                pass
                slopesl.append(np.nan)
                corrs.append(np.nan)
                corr_r.append(np.nan)
                numberMeas.append(np.nan)
                Swslist.append(np.nan)
                Fclist.append(np.nan)
            
            
            conv_fact = 10**6/(12*24*60*60)#gC /m^2 / day -> mu mol/m^2 /s
            if len(dfStationmerge_night[(dfStationmerge_night[tempvar] <= T_board[Ti][1])&(dfStationmerge_night[tempvar] > T_board[Ti][0])].swc1) >= minMeasnum: 
                slopeLTER, interceptLTER, correlLTER, pLTER, perrLTER = linregress(dfStationmerge_night[(dfStationmerge_night[tempvar] <= T_board[Ti][1])&(dfStationmerge_night[tempvar] > T_board[Ti][0])].swc1.astype(np.float64), 
                                                                (dfStationmerge_night[(dfStationmerge_night[tempvar] <= T_board[Ti][1])&(dfStationmerge_night[tempvar] > T_board[Ti][0])].rh.astype(np.float64)+dfStationmerge_night[(dfStationmerge_night[tempvar] <= T_board[Ti][1])&(dfStationmerge_night[tempvar] > T_board[Ti][0])].ra.astype(np.float64))*conv_fact)
                slopeLrh, interceptLrh, correlLrh, pLrh, perrLrh = linregress(dfStationmerge_night[(dfStationmerge_night[tempvar] <= T_board[Ti][1])&(dfStationmerge_night[tempvar] > T_board[Ti][0])].swc1.astype(np.float64), 
                                                                (dfStationmerge_night[(dfStationmerge_night[tempvar] <= T_board[Ti][1])&(dfStationmerge_night[tempvar] > T_board[Ti][0])].rh.astype(np.float64))*conv_fact)
                slopeLra, interceptLra, correlLra, pLra, perrLra = linregress(dfStationmerge_night[(dfStationmerge_night[tempvar] <= T_board[Ti][1])&(dfStationmerge_night[tempvar] > T_board[Ti][0])].swc1.astype(np.float64), 
                                                                (dfStationmerge_night[(dfStationmerge_night[tempvar] <= T_board[Ti][1])&(dfStationmerge_night[tempvar] > T_board[Ti][0])].ra.astype(np.float64))*conv_fact)
                if LPJParam == 'TER':
                    slopeL = slopeLTER
                    perrL = perrLTER
                elif  LPJParam == 'rh':
                    slopeL = slopeLrh
                    perrL = perrLrh
                elif  LPJParam == 'ra':
                    slopeL = slopeLra
                    perrL = perrLra
                if plotdataset == 'LPJ':
                    savestr = plotdataset+'_'+tempvar+'_'+LPJParam
                    ax5.plot(stid +0.5, slopeL, marker = 'x', color = T_board_color[Ti], markersize = 4)
                    ax5.errorbar(stid +0.5, slopeL, perrL, color = T_board_color[Ti])
            

                slopesLPJlTER.append(slopeLTER)
                slopesLPJlrh.append(slopeLrh)
                slopesLPJlra.append(slopeLra)


            else:
                slopesLPJlTER.append(np.nan)
                slopesLPJlrh.append(np.nan)
                slopesLPJlra.append(np.nan)
                

            if stid == len(stationlist)-1:
                ax5.plot([np.nan],[np.nan],marker='x', ls = '', color = T_board_color[Ti], markersize = 4, label = 'T=['+str(T_board[Ti][0])+','+str(T_board[Ti][1])+']')
                
              
        

if Figure2:
    dfTempCorr = pd.DataFrame(data={'Fullname':Stationfullnamecorrlist,
                                            'SlopeLPJTER':slopesLPJlTER,
                                            'SlopeLPJrh':slopesLPJlra,
                                            'SlopeLPJrh':slopesLPJlra,
                                            'SlopeOz':slopesl,
                                            'CorrOzFlux':corr_r,
                                            'Corr_r2OzFlux':corrs,
                                            'station':Stationcorrlist,
                                            'Temp':templist,
                                            'Number':numberMeas,
                                            'MeanOzFluxFc':Fclist,
                                            'MeanOzFluxSws':Swslist})
    dfTempCorr.to_csv(savepath+'/dfCorr_'+tempvar+'.csv')
    dfTempCorr.to_pickle(savepath+'/dfCorr_'+tempvar+'.pkl')

    if plotdataset == 'LPJ':
        ax5.set_ylabel('Sensitivity '+LPJParam+'/Sws\n'+ r'$\rm [(\mu mol/m^2/s)/(m^3/m^3)]$')
    else:
        ax5.set_ylabel('Sensitivity TER/Sws\n'+ r'$\rm [(\mu mol/m^2/s)/(m^3/m^3)]$')
    ax5.set_xticks(ticks = np.array(range(len(stationlist)))+0.5)
    xticklabelslist = [str(i) + '. '+stationlist_fullnames[np.where(np.array(stationlist) == stationnamelist[i-1])[0][0]] for i in range(1, len(stationnamelist)+1)]
    ax5.set_xticklabels(xticklabelslist, rotation = 90,fontsize=7)
    #ax5.set_xlabel('Soil Moisture [%]')
    ax5.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax5.grid(True,which = 'major', axis='y',zorder = 0)
    fig5.savefig(savepath+'/SensitivityperStation_'+savestr+'.png',bbox_inches='tight',dpi = 400)


if FigureA5:
    ax4.set_ylim(-10,80)

    ax4.set_ylabel('Sensitivity TER/Sws \n'+ r'$\rm [(\mu mol/m^2/s)/(m^3/m^3)]$')
    ax4.set_xticks(ticks = np.array(range(len(stationlist)))+0.5)
    xticklabelslist = [str(i) + '. '+stationlist_fullnames[np.where(np.array(stationlist) == stationnamelist[i-1])[0][0]] for i in range(1, len(stationnamelist)+1)]
    ax4.set_xticklabels(xticklabelslist, rotation = 90,fontsize=7)
    ax4.text(1,72, textlist[TiA5], fontsize = 12)
    ax4.set_title(str(T_board[TiA5][0])+'°C - '+str(T_board[TiA5][1])+'°C'+', p='+str(anovalist[TiA5]))
    ax4.grid(True,which = 'major', axis='y',zorder = 0)
    cb = ax4.pcolorfast((0,len(stationlist)), ax4.get_ylim(),
                np.array(BGcorrs)[np.newaxis],
                cmap='binary', alpha=0.3,vmin=0,vmax=0.5)
    #plt.colorbar(cb, ax=ax4, label = 'R**2 of linear fit', orientation='horizontal', shrink = 0.7 , pad = 0.5)

    fig4.savefig(savepath+'/SensitivityperStation_OzFlux_'+str(T_board[TiA5][0])+'_'+str(T_board[TiA5][1])+'_5degInt.png',bbox_inches='tight',dpi = 300)



#Figure A1: Soil Moisture Boxplot
if True:
    #names = drystations + wetstations
    dfStation0 = pd.read_pickle(savepath+'/StationParam.pkl')
    dfCorr = pd.read_pickle(savepath+'/dfCorr_TsoilOrig.pkl')
    dfStation0 = pd.merge(dfStation0,dfCorr[(dfCorr.Temp == 9999)][['station','MeanOzFluxSws']], on ='station', how = 'left')
    dfStation0.rename(columns = {'MeanOzFluxSws':'Sws'}, inplace = True)
    dfStation0.sort_values('Sws', inplace = True)
    xticklabelslist = [str(i+1) + '. '+dfStation0.Fullname.values[i] for i in range(0, len(dfStation0.Fullname))]
    fig, ax = plt.subplots(figsize = (12,5))

    for i in range(len(dfStation0.station)):
        dfStation, dfStation_night, lats, longs = ReadAndQFilterOzFluxDaily(dfStation0.station.values[i])
        dfStation_nightT = dfStation_night[(dfStation_night.Ts >= 20)&(dfStation_night.Ts <= 25)]
        plt.boxplot(dfStation_night.Sws,positions=np.array([i])-0.15)
        if len(dfStation_nightT.Sws) > 100:
            plt.boxplot(dfStation_nightT.Sws,positions=np.array([i])+0.15,boxprops=dict(color='green'),whiskerprops=dict(color='green'),capprops=dict(color='green'),showfliers=False)
        #plt.boxplot(dfStation_night.Sws,positions=np.array([i]),showmeans=True, meanline=True,whis=[5,95],showfliers=False)
        if i == 0:
            plt.plot([np.nan],[np.nan],marker='s',ls='', color = 'black', label = 'all temperatures')
            plt.plot([np.nan],[np.nan],marker='s',ls='', color = 'green',label = '20°C - 25°C')
        ax.grid(True,which='major',axis='y')
        ax.set_xticks(ticks = np.array(range(len(dfStation0.station))))
        
        ax.set_xticklabels(xticklabelslist, rotation = 90,fontsize=10)
        ax.set_yticks(ticks = np.array(range(11))/10)
        ax.set_yticklabels(range(0,110,10),fontsize=10)
        ax.set_ylabel('Soil moisture [%]')
    ax.legend()
    plt.savefig(savepath+"/FigureA1.png", dpi=300, bbox_inches = "tight")
    plt.savefig(savepath+"/FigureA1.eps", dpi=300, bbox_inches = "tight")
    plt.savefig(savepath+"/FigureA1.pdf", dpi=300, bbox_inches = "tight",format = 'pdf')



# ANOVA calculations
if True:
    df = pd.read_pickle(savepath+'/dfCorr_TsoilOrig.pkl')
    df = df[(df.Temp != 9999)]
    
    df.insert(0, column = 'type', value = df.apply(lambda x: 'wet' if x.station in wetstations else 'dry', axis = 1))
    
    for x,var in enumerate(['SlopeLPJTER', 'SlopeLPJrh', 'SlopeOz', 'CorrOzFlux', 'Corr_r2OzFlux']):
        statsl = []
        pvall = []
        # all T
        typevar = 'type'
        classvar1 = 'wet'
        classvar2 = 'dry'
        res0 = scipy.stats.f_oneway(df[(df[typevar] == classvar1)][var].dropna().values,df[(df[typevar] == classvar2)][var].dropna().values)
        statsl.append(res0.statistic)
        pvall.append(res0.pvalue)      
        del res0          
        # single T
        for T in range(0,40,5):
            res = scipy.stats.f_oneway(df[(df[typevar] == classvar1)&(df.Temp==T)][var].dropna().values,df[(df[typevar] == classvar2)&(df.Temp==T)][var].dropna().values)
            statsl.append(res.statistic)
            pvall.append(res.pvalue)
            del res
        dfANOVA0 = pd.DataFrame(data={'Temp':[9999]+list(range(0,40,5)),var+'_stat':statsl,var+'_pval':pvall})
        if x == 0:
            dfANOVA = dfANOVA0.copy()
        else:
            dfANOVA = pd.merge(dfANOVA,dfANOVA0,how='left',on='Temp')
    
    dfANOVA.to_pickle(savepath+'/AnovaResults_TsoilOrig_'+typevar+'_'+classvar1+'_'+classvar2+'.pkl')
    dfANOVA.to_csv(savepath+'/AnovaResults_TsoilOrig_'+typevar+'_'+classvar1+'_'+classvar2+'.csv')
    
    
#Figure 4 Box plot R² wet stations and dry stations
if True:
    df = pd.read_pickle(savepath+'/dfCorr_TsoilOrig.pkl')
    
    fig3, ax3 = plt.subplots(1,2, figsize = (12,5))
    #ax1.plot(7*np.ones(len(df[(df['station'].isin(wetstations))&(df.Temp == 30)].Temp)),
    #                df[(df['station'].isin(wetstations))&(df.Temp == 30)].Corr_r2,ls='',marker = 'o',color = 'black',markerfacecolor='none') 
    ax3[0].plot(8*np.ones(len(df[(df['station'].isin(wetstations))&(df.Temp == 35)].Temp)),
                    df[(df['station'].isin(wetstations))&(df.Temp == 35)].Corr_r2OzFlux,ls='',marker = 'o',color = 'black',markerfacecolor='none') 
    ax3[0].plot(1*np.ones(len(df[(df['station'].isin(wetstations))&(df.Temp == 0)].Temp)),
                    df[(df['station'].isin(wetstations))&(df.Temp == 0)].Corr_r2OzFlux,ls='',marker = 'o',color = 'black',markerfacecolor='none') 
    df[df['station'].isin(wetstations)&(df.Temp <40)&~((df.Temp==35)&(df.Corr_r2OzFlux > 0))&~((df.Temp==0)&(df.Corr_r2OzFlux > 0))].boxplot(column='Corr_r2OzFlux',by='Temp',ax=ax3[0],showmeans=True, meanline=True)
    #for T in [0,5,10,15,20,25,30,35]:    
    #    print(df[df['station'].isin(wetstations)&(df.Temp ==T)].describe())
    
    ax3[0].text(0.23, 0.94, r'$\rm (a)~Wet~stations$', horizontalalignment='center', verticalalignment='center', transform=ax3[0].transAxes,fontsize=13)#, weight='bold')      
    ax3[0].set_xticks([0.5,1,2,3,4,5,6,7,8])#np.array(range(9))-0.5,[0,5,10,15,20,25,30,35])
    ax3[0].set_xticklabels(['','0-5','5-10','10-15','15-20','20-25','25-30','30-35','35-40'])
    
    ax3[0].set_ylabel(r'$\rm R^2$')
    ax3[0].set_xlabel('Temperature [°C]')
    ax3[0].set_ylim(-0.02,0.55)
    ax3[0].get_figure().suptitle("")
    ax3[0].set_title("")
    
    
    ax3[1].plot(8*np.ones(len(df[(df['station'].isin(drystations))&(df.Temp == 35)].Temp)),
                    df[(df['station'].isin(drystations))&(df.Temp == 35)].Corr_r2OzFlux,ls='',marker = 'o',color = 'black',markerfacecolor='none') 
    #df[(df['station'].isin(drystations))&(df.Temp<40)&~((df.Temp==30)&(df.Corr_r2 > 0))].boxplot(column='Corr_r2',by='Temp',ax=ax)
    df[(df['station'].isin(drystations))&(df.Temp<35)].boxplot(column='Corr_r2OzFlux',by='Temp',ax=ax3[1],showmeans=True, meanline=True)
    ax3[1].text(0.2, 0.94, r'$\rm (b)~Dry~stations$', horizontalalignment='center', verticalalignment='center', transform=ax3[1].transAxes,fontsize=13)#, weight='bold')      
    #for T in [0,5,10,15,20,25,30,35]:    
    #    print(df[(df['station'].isin(drystations))&(df.Temp==T)].describe())
    ax3[1].set_xticks([0.5,1,2,3,4,5,6,7,8])#np.array(range(9))-0.5,[0,5,10,15,20,25,30,35])
    ax3[1].set_xticklabels(['','0-5','5-10','10-15','15-20','20-25','25-30','30-35','35-40'])
    
    ax3[1].set_ylabel(r'$\rm R^2$')
    ax3[1].set_xlabel('Temperature [°C]')
    ax3[1].get_figure().suptitle("")
    ax3[1].set_title("")
    
    ax3[1].set_ylim(-0.02,0.55)
    ax3[1].set_xlim(0.5, 8.5)
    plt.savefig(savepath+"/Figure4_Boxplot_R2_dry_humid_midinterval.png", dpi=300, bbox_inches = "tight")
    plt.savefig(savepath+"/Figure4_Boxplot_R2_dry_humid_midinterval.eps", dpi=300, bbox_inches = "tight")
    plt.savefig(savepath+"/Figure4_Boxplot_R2_dry_humid_midinterval.pdf", dpi=300, bbox_inches = "tight",format='pdf')


#Figure 1: Station Map
if True:
    dsAI_AU = xr.open_dataset(savepath+"/ai_v3_yr_AU.nc")
    #dsAI_AU = dsAI.sel(latitude=slice(-10,-54),longitude=slice(112,155))
    dfStationParam = pd.read_csv(savepath+'/StationParam.csv')
    dfCorr = pd.read_pickle(savepath+'/dfCorr_TsoilOrig.pkl')
    dfStationParam = pd.merge(dfStationParam,dfCorr[(dfCorr.Temp == 9999)][['station','MeanOzFluxSws']], on ='station', how = 'left')
    dfStationParam.rename(columns = {'MeanOzFluxSws':'Sws'}, inplace = True)
    dfStationParam.sort_values('Sws', inplace = True)
    dfStationParam.reset_index(inplace = True)
    gdfStationParam = geopandas.GeoDataFrame(
        dfStationParam, geometry=geopandas.points_from_xy(dfStationParam.long, dfStationParam.lat),crs = 'EPSG:4326')
    gdfStationParam.insert(0,column='order',value=range(len(gdfStationParam.station)))

    annotatelist1= gdfStationParam.apply(lambda x: str(x.order + 1) if (x.regs not in ['N','SE'] or x.station in ['Warra','AliceSpringsMulga','TiTreeEast','CumberlandMelaleuca','CumberlandPlain','YarCon','YarIrr']) else '',axis = 1)
    annotatelist2= gdfStationParam.apply(lambda x: str(x.order + 1) if (x.regs == 'N' and x.station not in ['AliceSpringsMulga','TiTreeEast']) else '',axis = 1)
    annotatelist3= gdfStationParam.apply(lambda x: str(x.order + 1) if (x.regs == 'SE' and x.station not in ['Warra','CumberlandMelaleuca','CumberlandPlain','YarCon','YarIrr']) else '',axis = 1)
    #offsetlistx =[1, 2 , 3, 4, 5,  6,  7,  8,  9,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40]]
    offsetlistx = [1,0.3,-2, 1, 1,0.3,0.3, -2,0.3,-0.4, 0.3,-1.5,   1,-1.2,-0.9,   1,-1.5,   1,-0.8,   1,-0.8, 0.1,-0.9,   1, 0.1,-0.8,   1, 0.1,-0.7,   1,-0.8,   1,   1, 0.2,-0.5,-0.7,  -3,-0.3, 0.2, 0.2]
    offsetlisty = [1, 0 , 0, 1,-2,0 ,-0.4, -2,-0.2,-0.6,  0,   3,-0.1,  0 , 0.3,  -2, 1.2,   0, 0.2,-3.8, 0.2, 0.2, 0.1,  -2,-0.4, 0.1,   1,-0.5,-0.4,-0.5, 0.5,  -2,   0,-0.1,  -2,-0.6,   0,-0.8, -0.9, 0.4] 

    fig = plt.figure(figsize=(10, 6))

    # Create a GridSpec object with 3 rows and 2 columns
    gs = GridSpec(2, 3, figure=fig)

    # Large subplot (spanning two columns and multiple rows)
    ax1 = fig.add_subplot(gs[0:2, 0:2])  # Takes the first two rows in the first column
    
    # Smaller subplot 1 (top right)
    ax2 = fig.add_subplot(gs[0, 2])  # Top row, second column
    
    # Smaller subplot 2 (bottom right)
    ax3 = fig.add_subplot(gs[1, 2])  # Bottom row, second column
    
    ax1.plot([np.nan],[np.nan], marker = 's',ls='',markersize = 5,color = (0.7,0.7,0.7), label='Arid (0.03 < AI < 0.2)')
    ax1.plot([np.nan],[np.nan], marker = 's',ls='',markersize = 5,color = (0.5,0.5,0.5), label='Semi-arid (0.2 < AI < 0.5)')
    ax1.plot([np.nan],[np.nan], marker = 's',ls='',markersize = 5,color = (0.3,0.3,0.3), label='Dry sub-humid (0.5 < AI < 0.65)')
    ax1.plot([np.nan],[np.nan], marker = 's',ls='',markersize = 5,color = 'black', label='Humid (AI > 0.65)')
    pl1 = dsAI_AU.AI.where((dsAI_AU.AI>0)).plot(ax = ax1, levels=np.array([0.03,0.2,0.5,0.65]),colors=[(0.9,0.9,0.9),(0.7,0.7,0.7),(0.5,0.5,0.5),(0.3,0.3,0.3),'black'],add_colorbar=False)#, vmin = 0, vmax = 100)
    gdfStationParam[(gdfStationParam.Sws<0.12)].plot(column = 'Sws',ax = ax1,marker = 'x',markersize = 100, cmap = 'Spectral',vmin = gdfStationParam.Sws.min(),vmax = gdfStationParam.Sws.max(),legend=True,legend_kwds={"label": "Soil Moisture [%]", "orientation": "horizontal","shrink":0.6})
    gdfStationParam[(gdfStationParam.Sws>=0.12)].plot(column = 'Sws',ax = ax1,marker = '+',markersize = 100, cmap = 'Spectral',vmin = gdfStationParam.Sws.min(),vmax = gdfStationParam.Sws.max())#,legend=True,legend_kwds={"label": "Soil Moisture [%]", "orientation": "horizontal","shrink":0.6})
    # Add a rectangle highlighting the inset area (optional)
    rect = Rectangle((128, -18), 7, 7, facecolor="none", edgecolor="red", linestyle="--")
    ax1.add_patch(rect)
    rect = Rectangle((139, -40), 11, 7, facecolor="none", edgecolor="blue", linestyle="--")
    ax1.add_patch(rect)
    for i in range(len(annotatelist1)):
        ax1.annotate(annotatelist1[i],(gdfStationParam.long.values[i],gdfStationParam.lat.values[i]),xytext=(gdfStationParam.long.values[i]+offsetlistx[i],gdfStationParam.lat.values[i]+offsetlisty[i]),color = 'black')

    ax1.legend(loc=3)
    ax1.set_ylim(-50,-10)
    
    dsAI_AU.AI.where((dsAI_AU.AI>0)).plot(ax = ax2, levels=np.array([0.03,0.2,0.5,0.65]),colors=[(0.9,0.9,0.9),(0.7,0.7,0.7),(0.5,0.5,0.5),(0.3,0.3,0.3),'black'],add_colorbar=False)#, vmin = 0, vmax = 100)
    #gdfStationParam.plot(column = 'Sws',ax = ax2,marker = 'x',markersize = 100, cmap = 'Spectral')
    gdfStationParam[(gdfStationParam.Sws<0.12)].plot(column = 'Sws',ax = ax2,marker = 'x',markersize = 100, cmap = 'Spectral',vmin = gdfStationParam.Sws.min(),vmax = gdfStationParam.Sws.max())#,legend=True,legend_kwds={"label": "Soil Moisture [%]", "orientation": "horizontal","shrink":0.6})
    gdfStationParam[(gdfStationParam.Sws>=0.12)].plot(column = 'Sws',ax = ax2,marker = '+',markersize = 100, cmap = 'Spectral',vmin = gdfStationParam.Sws.min(),vmax = gdfStationParam.Sws.max())#,legend=True,legend_kwds={"label": "Soil Moisture [%]", "orientation": "horizontal","shrink":0.6})
    rect = Rectangle((128, -18), 7, 7, facecolor="none", edgecolor="red", linestyle="--")
    ax2.add_patch(rect)
    ax2.set_ylim(-18.1,-10.9)
    ax2.set_xlim(127.9, 135.1)
    for i in range(len(annotatelist1)):
        ax2.annotate(annotatelist2[i],(gdfStationParam.long.values[i],gdfStationParam.lat.values[i]),xytext=(gdfStationParam.long.values[i]+offsetlistx[i],gdfStationParam.lat.values[i]+offsetlisty[i]),color = 'black')

    dsAI_AU.AI.where((dsAI_AU.AI>0)).plot(ax = ax3, levels=np.array([0.03,0.2,0.5,0.65]),colors=[(0.9,0.9,0.9),(0.7,0.7,0.7),(0.5,0.5,0.5),(0.3,0.3,0.3),'black'],add_colorbar=False)#, vmin = 0, vmax = 100)
    #gdfStationParam.plot(column = 'Sws',ax = ax3,marker = 'x',markersize = 100, cmap = 'Spectral')
    gdfStationParam[(gdfStationParam.Sws<0.12)].plot(column = 'Sws',ax = ax3,marker = 'x',markersize = 100, cmap = 'Spectral',vmin = gdfStationParam.Sws.min(),vmax = gdfStationParam.Sws.max())#,legend=True,legend_kwds={"label": "Soil Moisture [%]", "orientation": "horizontal","shrink":0.6})
    gdfStationParam[(gdfStationParam.Sws>=0.12)].plot(column = 'Sws',ax = ax3,marker = '+',markersize = 100, cmap = 'Spectral',vmin = gdfStationParam.Sws.min(),vmax = gdfStationParam.Sws.max())#,legend=True,legend_kwds={"label": "Soil Moisture [%]", "orientation": "horizontal","shrink":0.6})
    rect = Rectangle((139, -40), 11, 7, facecolor="none", edgecolor="blue", linestyle="--")
    ax3.add_patch(rect)
    ax3.set_ylim(-40.1,-32.9)#(-19,-10)
    ax3.set_xlim(138.9,150.1)#(128, 135)
    for i in range(len(annotatelist1)):
        ax3.annotate(annotatelist3[i],(gdfStationParam.long.values[i],gdfStationParam.lat.values[i]),xytext=(gdfStationParam.long.values[i]+offsetlistx[i],gdfStationParam.lat.values[i]+offsetlisty[i]),color = 'black')
    plt.tight_layout()
    plt.savefig(savepath+'/Figure1_Map.png',bbox_inches='tight',dpi=300)
    #plt.savefig(savepath+'/Figure1_Map.eps',bbox_inches='tight',dpi=300)
    #plt.savefig(savepath+'/Figure1_Map.pdf',bbox_inches='tight',dpi=300,format='pdf')




# Figure A3 analytical functions    
if True:
    def g_T(T):
        res = np.exp(308.56*(1/56.02 - 1/(T + 46.02)))
        return res

    def f_W_exp(W):
        res = (1-np.exp(-W))/(1-np.exp(-1))
        return res

    def k(tau, T, W):
        res = 1/12 * (1/tau) * g_T(T) * f_W_exp(W)
        return res

    def C(C_0, tau, T, W, t):
        res = C_0 * np.exp(-k(tau, T, W) * t)
        return res

    def rh(C_0, tau, T, W, t):
        res = 0.7*(C_0 - C(C_0, tau, T, W, t))
        return res

    tau_litter = 2.86
    tau_inter = 33.3
    tau_slow = 1000   

    tau = tau_litter

    SMarray = np.linspace(0,1,num=50)
    Tarray = np.linspace(5,50, num = 50)
    colorlist = ['#440154FF','#404788FF','#2D708EFF','#1F968BFF','#55C667FF','#95D840FF','#DCE319FF','#FDE725FF','orange']
            
    fig2, ax2 = plt.subplots(figsize=(5.5, 4))
    for i,sm in enumerate(np.linspace(0,0.8,num = 9)):
        print()
        ax2.plot(Tarray, rh(100,tau, Tarray, sm, 1), ls = '-', color = colorlist[8-i], label =str(int(sm*100))+'%')
    ax2.legend(ncol = 2)
    ax2.set_ylabel('rh [kgC/m²/day]')
    ax2.set_xlabel('Temperature [°C]')
    plt.savefig(savepath+'/FigureA3_AnalyticalFunc_Temp.png',bbox_inches='tight',dpi=300)
    
    fig2, ax3 = plt.subplots(figsize=(5.5, 4))
    for i,t in enumerate(np.linspace(5,45,num = 9)):
        print()
        ax3.plot(SMarray, rh(100,tau, t, SMarray, 1), ls = '-', color = colorlist[i], label =str(int(t))+'°C')
    ax3.legend(ncol = 2)
    ax3.set_ylabel('rh [kgC/m²/day]')
    ax3.set_xlabel('Soil Moisture [%]')
    plt.savefig(savepath+'/FigureA3_AnalyticalFunc_SoilM.png',bbox_inches='tight',dpi=300)

# Figure A11    
if True:
    
    dfStation = pd.read_pickle(savepath+'/StationParam.pkl')
    dfCorr = pd.read_pickle(savepath+'/dfCorr_TsoilOrig.pkl')
    dfStation = pd.merge(dfStation,dfCorr[(dfCorr.Temp == 9999)][['station','MeanOzFluxSws']], on ='station', how = 'left')
    dfStation.rename(columns = {'MeanOzFluxSws':'Sws'}, inplace = True)
    dfStation.sort_values('Sws', inplace = True)
    
    fig2, ax3 = plt.subplots(figsize=(7, 4))
    ax3.bar(dfStation.Fullname, dfStation.Litter_new)
    ax3.set_ylabel('Litter')
    xticklabelslist = [str(i+1) + '. '+dfStation.Fullname.values[i] for i in range(0, len(dfStation.Fullname))]
        
    ax3.set_xticklabels(xticklabelslist,rotation = 90,fontsize=7)
        
    plt.savefig(savepath+'/FigureA11_Litter_Barplot.png',bbox_inches='tight',dpi=300)
    plt.savefig(savepath+'/FigureA11_Litter_Barplot.eps',bbox_inches='tight',dpi=300)
    plt.savefig(savepath+'/FigureA11_Litter_Barplot.pdf',bbox_inches='tight',dpi=300,format='pdf')
    

