# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from astropy.convolution import Gaussian2DKernel, convolve
import plotly.graph_objects as go
import plotly

savepath = '' #enter path

df = pd.read_pickle(savepath+"/dfCorr_TsoilOrig.pkl")
dfSws = df[(df.Temp == 9999)]
df = df[(df.Temp < 40)]

dfSm = pd.read_csv(savepath+"/StationParam.csv")
dfSm = pd.merge(dfSm, dfSws[['station','MeanOzFluxSws']],on='station',how='left')
dfSm.sort_values('MeanOzFluxSws',inplace=True)
dfSm.index = dfSm.station

dfSm.Fullname = ["1. Ti Tree East", "2. Calperum", "3. Gingin", "4. Boyagin", 
                 "5. Alice Springs M.", "6. R. D. Melon Farm", 
                 "7. Daly River U.", "8. Collie", "9. Dry River", "10. Litchfield", 
                 "11. Daly River P.", "12. Cumberland M.", "13. Arcturus", 
                 "14. Howard Springs", "15. Howard Springs U.", "16. Ridgefield", 
                 "17. Cumberland P.", "18. Great Western W.", "19. Yanco", "20. Yarramundi Con.", 
                 "21. Gatum Pasture", "22. Riggs Creek", "23. Whroo", 
                 "24. Yarramundi Irr.", "25. Adelaide River", "26. Wombat", 
                 "27. Cape Tribulation", "28. Nimmo", "29. Otway", "30. Cow Bay", 
                 "31. Tumbarumba", "32. Robson Creek", "33. Mitchell Grass R.", "34. Sturt Plains", 
                 "35. Warra", "36. Wallaby Creek", "37. Samford", "38. Dargo", 
                 "39. Alpine Peatland", "40. Fogg Dam"]

Matrix2 = df.groupby(['station','Temp'], sort=False)['SlopeOz'].sum().unstack('Temp')
Matrix2 = Matrix2.replace(0,np.nan)
Matrix2 = Matrix2.iloc[::-1]

Colors2 = df.groupby(['station','Temp'], sort=False)['Corr_r2OzFlux'].sum().unstack('Temp')
Colors2 = Colors2.replace(0,np.nan)
Colors2 = Colors2.iloc[::-1]

kernel = Gaussian2DKernel(x_stddev=1,y_stddev=2)
astropy_conv = convolve(Matrix2, kernel, boundary="extend")
astropy_convDF = pd.DataFrame(data= astropy_conv, index = Matrix2.index, columns = Matrix2.keys())


Colors5astropy_conv = convolve(Colors2, kernel, boundary="extend")
Colors5 = pd.DataFrame(data= Colors5astropy_conv, index = Colors2.index, columns = Colors2.keys())

datamatrix = astropy_convDF

x_vals = np.tile(datamatrix[~np.isnan(Matrix2)].columns, len(datamatrix[~np.isnan(Matrix2)].index ))+15# Repeat columns for each row
y_vals = np.repeat(datamatrix[~np.isnan(Matrix2)].index, len(datamatrix[~np.isnan(Matrix2)].columns))  # Repeat index for each column
z_vals = datamatrix[~np.isnan(Matrix2)].values.flatten()  # Flatten to 1D

x_positions = [ 0,5,10,15,20,25,30,35,40,45,50]
x_labels = [ "0%","50%", "100%", "0°-5°", "5°-10°", "10°-15°", "15°-20°","20°-25°","25°-30°","30°-35°","35°-40°"]


wetstations = ['FoggDam', 'fallscreek', 'Dargo', 'Samford', 'Wallaby', 
       'SturtPlains','Warra', 'Longreach', 'Robson', 'Tumbarumba', 'CowBay', 'Otway',
       'Nimmo', 'CapeTribulation', 'WombatStateForest', 'AdelaideRiver',
       'YarIrr', 'Whroo', 'Riggs', 'GatumPasture', 'YarCon', 'Yanco', 'GWW',
       'CumberlandPlain', 'Ridgefield']
drystations= ['HowardUnderstory', 'HowardSprings',
       'Emerald', 'CumberlandMelaleuca', 'DalyPasture', 'Litchfield',
       'DryRiver', 'Collie', 'DalyUncleared', 'RDMF', 'AliceSpringsMulga',
       'Boyagin1', 'Gingin', 'Calperum', 'TiTreeEast']


fig = go.Figure()
fig.add_scatter3d(x= x_vals, y=y_vals, z=z_vals,mode='markers', 
    marker=dict(size=5, color='black'),showlegend=False)#, color = 'black')

fig.add_surface(x= datamatrix.T[wetstations].T.columns +15, y=datamatrix.T[wetstations].T.index, z=datamatrix.T[wetstations].T,surfacecolor=Colors5.T[wetstations].T,colorbar=dict(title="R^2"),cmin =0,cmax=0.30)
fig.add_surface(x= datamatrix.T[drystations].T.columns +15, y=datamatrix.T[drystations].T.index, z=datamatrix.T[drystations].T,surfacecolor=Colors5.T[drystations].T,colorbar=dict(title="R^2"),cmin =0,cmax=0.30, showscale=False)

fig.add_scatter3d(x=dfSm.T[wetstations].T.MeanOzFluxSws*10, y=dfSm.T[wetstations].T.station , z=np.zeros(len(dfSm.T[wetstations].T.MeanOzFluxSws)), mode='lines', 
    line=dict(color='black', width=5), showlegend=False)
fig.add_scatter3d(x=dfSm.T[drystations].T.MeanOzFluxSws*10, y=dfSm.T[drystations].T.station , z=np.zeros(len(dfSm.T[drystations].T.MeanOzFluxSws)), mode='lines', 
    line=dict(color='black', width=5), showlegend=False)
fig.add_surface(x= [[0,10],[0,10]], y=[['TiTreeEast','TiTreeEast'],['FoggDam','FoggDam']], z=[[0,0],[0,0]],colorscale=[[0,'lightblue'],[1,'lightblue']],opacity=0.5, showscale=False)

fig.update_layout(plotly.graph_objs.Layout(#autosize=False,width=800, height=900,
                                                    scene = dict(xaxis=dict(
        tickmode="array",
        title="Sws [%]           Temperature [°C]",
        tickvals=x_positions,
        ticks="outside",
        ticktext=x_labels),yaxis=dict(title='',tickmode='array', 
                    tickvals = dfSm.station,
                    ticktext = dfSm.Fullname, ticks="outside" ),xaxis_range=(-5,52),zaxis_title='Sensitivity TER/Sws<br>[(umol/m²/s)/(m³/m³)]',zaxis_range=(0,43)),                                                       
                                                    margin=dict(l=15, r=15, b=50, t=0)))
fig.update_layout(
    scene=dict(
        aspectratio=dict(x=1, y=1.5, z=1),
        yaxis=dict(nticks=60,tickfont=dict(size=10))  ))

fig.write_html(savepath+"/Metz2025_3DFig_new.html")
