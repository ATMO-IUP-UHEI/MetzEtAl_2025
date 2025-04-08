#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 19:48:21 2024
"""

import numpy as np
import pandas as pd

savepath = ''

#path to LPJ files
BinFilePathList = [savepath + "/AllOzFlux_p1_default_wTs/binary_outputs/",
                   savepath +  "/AllOzFlux_p2_default_wTs/binary_outputs/"]
savestr = 'AllOzFlux_default_wTs'
NumberStationFiles = np.array([19,19])
NumberStation = NumberStationFiles.sum()


stationlist = ['BeaconFarm','AD','Cumberland_YarCon_YarIrr','Emerald',
               'Yanco','Robson','Riggs','Longreach',
               'Otway','Digby','Calperum','ASM',
               'DalyPasture_SturtPlains','TTE','DryRiver','RDMF',
               'DalyUncleared_DalyRegrowth','AdelaideRiver','Howard','Litchfield',
               'Ridgefield','Boyagin',
               'CowBay','FoggDam','GWW','Gingin','Tumbarumba',
                'Warra','Whroo','WombatStateForest','fallscreek',
                'CapeTribulation','Collie','Dargo','GatumPasture',
                'Nimmo','Samford','Wallaby']

#monthly
#endyear = 2023
#startyear = startyear - nspin_transient = 1901 - 41
#daily
#startyear = 1990
dates = pd.date_range(start="1990-01-01",end="2023-12-31",freq = 'D')
dates = dates[(dates.month != 2)|(dates.day != 29)]
#monthly soil moisture starts at 1700!
mdates = pd.date_range(start="1700-01-01",end="2023-12-31",freq = 'MS')
jdates = pd.date_range(start="1700-01-01",end="2023-12-31",freq = 'Y')

stationdata = []
Mstationdata = []
Ystationdata = []


for i in range(len(BinFilePathList)):
    
    Dnpp = np.fromfile(BinFilePathList[i] + 'dnpp_00000.bin', dtype = np.float32)
    DRh = np.fromfile(BinFilePathList[i] + 'drh_00000.bin', dtype = np.float32)
    Dgpp = np.fromfile(BinFilePathList[i] + 'dgpp_00000.bin', dtype = np.float32)
    Dppt = np.fromfile(BinFilePathList[i] + 'dppt_00000.bin', dtype = np.float32)
    Mswc1 = np.fromfile(BinFilePathList[i] + 'mswc1_00000.bin', dtype = np.float32)
    Mswc2 = np.fromfile(BinFilePathList[i] + 'mswc2_00000.bin', dtype = np.float32)
    Dswc1 = np.fromfile(BinFilePathList[i] + 'dswc1_00000.bin', dtype = np.float32)
    Dswc2 = np.fromfile(BinFilePathList[i] + 'dswc2_00000.bin', dtype = np.float32)
    Dtair = np.fromfile(BinFilePathList[i] + 'dtair_00000.bin', dtype = np.float32)
    gT = np.fromfile(BinFilePathList[i] + 'dgtemp_soil_00000.bin', dtype = np.float32)
    Tsoil = np.fromfile(BinFilePathList[i] + 'dsoiltemp_00000.bin', dtype = np.float32)
    Jlitter = np.fromfile(BinFilePathList[i] + 'litc_00000.bin', dtype = np.float32)
    Jsoil = np.fromfile(BinFilePathList[i] + 'soilc_00000.bin', dtype = np.float32)

    for numSt in range(NumberStationFiles[i]):
        
        startPos = numSt
        PosStep = NumberStationFiles[i]
        
        stationdata.append(pd.DataFrame(data={'Date':dates,
                                              'rh':DRh[range(startPos,len(DRh),PosStep)],
                                              'gpp':Dgpp[range(startPos,len(Dgpp),PosStep)],
                                              'npp':Dnpp[range(startPos,len(Dnpp),PosStep)],
                                              'ppt':Dppt[range(startPos,len(Dppt),PosStep)],
                                              'swc1':Dswc1[range(startPos,len(Dswc1),PosStep)],
                                              'swc2':Dswc2[range(startPos,len(Dswc2),PosStep)],
                                              'tair':Dtair[range(startPos,len(Dtair),PosStep)],
                                              'gTsoil':gT[range(startPos,len(gT),PosStep)],
                                              'TsoilOrig':Tsoil[range(startPos,len(gT),PosStep)],
                                              'Year':dates.year,
                                              'Month':dates.month,
                                              'Day':dates.day}))
        stationdata[-1].insert(0, column = 'Tsoilcalc',value=(1/(1/56.02-(np.log(stationdata[-1].gTsoil)/308.56))+227.13 - 273.15))
        Mstationdata.append(pd.DataFrame(data={'Date':mdates,
                                              'swc1':Mswc1[range(startPos,len(Mswc1),PosStep)],
                                              'swc2':Mswc2[range(startPos,len(Mswc2),PosStep)],
                                              'Year':mdates.year,
                                              'Month':mdates.month}))
        Ystationdata.append(pd.DataFrame(data={'Date':jdates,
                                              'litter':Jlitter[range(startPos,len(Jlitter),PosStep)],
                                              'soc':Jsoil[range(startPos,len(Jsoil),PosStep)],
                                              'Year':jdates.year}))

for numSt in range(NumberStation):
    stationdata[numSt].to_pickle(savepath + '/'+stationlist[numSt]+'_'+savestr+'_v0.pkl')
    Ystationdata[numSt].to_pickle(savepath + '/'+stationlist[numSt]+'_'+savestr+'_Annual_v0.pkl')
        
