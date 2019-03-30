# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 10:56:35 2019

@author: yuxiang
"""

import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# read data and calculate log return
expiry = pd.read_csv('Expiry_Dates.csv',header=None,names=['symbol','expiry'])
expiry['expiry'] = pd.to_datetime(expiry['expiry'])
data = pd.read_csv('data.csv',index_col=0)
data.date = pd.to_datetime(data.date)
data=data.merge(expiry,how='left',left_on='m_localSymbol',right_on='symbol')
data['logRet_daily'] = np.log(data['close']/data['open'])/(dt.timedelta(hours=0.5)/dt.timedelta(hours=7))  # 16:00-9:30 + 00:30

# deal with duplicated data
# some data of same symbols at the same time have different quotes, take everage of their return
# 84 records are duplicate
duplicated_records = data[data.duplicated(subset=['date','m_localSymbol'])]
data['mean'] = data.groupby(['date','m_localSymbol'])['logRet_daily'].transform(np.mean)
data = data.drop(duplicated_records.index)
data = data.drop(columns='logRet_daily')
data.rename(columns={'mean':'logRet_daily'},inplace=True)

# =============================================================================
# SPX and VIX
# =============================================================================

# distribution of SPX return
SPX=data[data['m_localSymbol']=='SPX']
plt.hist(SPX['logRet_daily'],bins=40)
plt.title("Distribution of SPX")
plt.xlabel("log return") 
plt.ylabel("Frequency") 
plt.show()

percent=[]
for i in [x/1000 for x in range(-30,0,5)]:
    p=len(SPX[SPX['logRet_daily']<i])/len(SPX)
    percent.append((i,p))

# take -0.02 as the threshold of selloff
TH=-0.02
selloff = SPX[SPX['logRet_daily']<TH][['date','logRet_daily']]
selloff.rename(columns={'logRet_daily':'SPX'},inplace=True)
selloff['const']=1

# merge no lag VIX data to selloff
selloff = selloff.merge(data[data.m_localSymbol=='VIX'][['date','logRet_daily']],how='inner')
selloff.rename(columns={'logRet_daily':'VIX'},inplace=True)
# merge lag 30mins data to selloff
temp_lag1 = data.copy() 
temp_lag1.date = temp_lag1.date - dt.timedelta(hours=0.5)
selloff = selloff.merge(temp_lag1[temp_lag1['m_localSymbol']=='VIX'][['date','logRet_daily']],how='left')
selloff.rename(columns={'logRet_daily':'VIX_lag1'},inplace=True)
# merge lag 60mins data to selloff
temp_lag2 = data.copy() 
temp_lag2.date = temp_lag2.date - dt.timedelta(hours=1)
selloff = selloff.merge(temp_lag2[temp_lag2['m_localSymbol']=='VIX'][['date','logRet_daily']],how='left')
selloff.rename(columns={'logRet_daily':'VIX_lag2'},inplace=True)
# merge lag 90mins data to selloff
temp_lag3 = data.copy() 
temp_lag3.date = temp_lag3.date - dt.timedelta(hours=1.5)
selloff = selloff.merge(temp_lag3[temp_lag3['m_localSymbol']=='VIX'][['date','logRet_daily']],how='left')
selloff.rename(columns={'logRet_daily':'VIX_lag3'},inplace=True)

# selloff.isna().sum()

# correlation
temp=selloff.dropna()
SPX_VIX_corr = np.corrcoef([temp['SPX'],temp['VIX'],temp['VIX_lag1'],
                        temp['VIX_lag2'],temp['VIX_lag3']])
SPX_VIX_corr_table = pd.DataFrame(data=SPX_VIX_corr,index=['SPX','VIX','VIX_lag1','VIX_lag2','VIX_lag3'],columns=['SPX','VIX','VIX_lag1','VIX_lag2','VIX_lag3'])
    
# scatter plots
plt.figure(figsize=(8,6))
plt.subplot(2,2,1)
plt.scatter(selloff.SPX,selloff.VIX)
plt.xlabel('SPX return')
plt.ylabel('VIX return')
plt.title('SPX & VIX')

plt.subplot(2,2,2)
plt.scatter(selloff.dropna().SPX,selloff.dropna().VIX_lag1)
plt.xlabel('SPX return')
plt.ylabel('VIX lag1 return')
plt.title('SPX & VIX_lag1')

plt.subplot(2,2,3)
plt.scatter(selloff.dropna().SPX,selloff.dropna().VIX_lag2)
plt.xlabel('SPX return')
plt.ylabel('VIX lag2 return')
plt.title('SPX & VIX_lag2')

plt.subplot(2,2,4)
plt.scatter(selloff.dropna().SPX,selloff.dropna().VIX_lag3)
plt.xlabel('SPX return')
plt.ylabel('VIX lag3 return')
plt.title('SPX & VIX_lag3')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
    
# no lag
X=selloff[['const','SPX']].to_numpy()
y=selloff['VIX']
reg = sm.OLS(y, X).fit()
print(reg.summary())

len(selloff[selloff['VIX']>0])/len(selloff)

theta = np.linspace(-0.2, 0, 50, endpoint=True)
plt.scatter(selloff.SPX,selloff.VIX)
plt.plot(theta,reg.params[0]+reg.params[1]*theta,color='r')
plt.xlabel('SPX return')
plt.ylabel('VIX return')
plt.title('SPX & VIX')
plt.show()
#plt.plot(selloff.date,selloff.SPX)
#plt.plot(selloff.date,selloff.VIX)

# lag 30mins
X_lag1=selloff.dropna(subset=['VIX_lag1'])[['const','SPX']].to_numpy()
y_lag1=selloff.dropna(subset=['VIX_lag1'])['VIX_lag1']
reg_lag1 = sm.OLS(y_lag1, X_lag1).fit()
print(reg_lag1.summary())

temp = selloff.dropna(subset=['VIX_lag1'])
len(temp[temp['VIX_lag1']>0])/len(temp)

# lag 60mins
X_lag2=selloff.dropna(subset=['VIX_lag2'])[['const','SPX']].to_numpy()
y_lag2=selloff.dropna(subset=['VIX_lag2'])['VIX_lag2']
reg_lag2 = sm.OLS(y_lag2, X_lag2).fit()
print(reg_lag2.summary())

temp = selloff.dropna(subset=['VIX_lag2'])
len(temp[temp['VIX_lag2']>0])/len(temp)

plt.scatter(selloff.SPX,selloff.VIX_lag2)
plt.plot(theta,reg_lag2.params[0]+reg_lag2.params[1]*theta,color='r')
plt.xlabel('SPX return')
plt.ylabel('VIX lag2 return')
plt.title('SPX & VIX_lag2')
plt.show()

# lag 90mins
X_lag3=selloff.dropna(subset=['VIX_lag3'])[['const','SPX']].to_numpy()
y_lag3=selloff.dropna(subset=['VIX_lag3'])['VIX_lag3']
reg_lag3 = sm.OLS(y_lag3, X_lag3).fit()
print(reg_lag3.summary())

temp = selloff.dropna(subset=['VIX_lag3'])
len(temp[temp['VIX_lag3']>0])/len(temp)

plt.scatter(selloff.SPX,selloff.VIX_lag3)
plt.plot(theta,reg_lag3.params[0]+reg_lag3.params[1]*theta,color='r')
plt.xlabel('SPX return')
plt.ylabel('VIX lag3 return')
plt.title('SPX & VIX_lag3')
plt.show()

# aggregate plot
theta = np.linspace(-0.2, 0, 50, endpoint=True)
plt.figure(figsize=(8,6))

plt.subplot(2,2,1)
plt.scatter(selloff.SPX,selloff.VIX)
plt.plot(theta,reg.params[0]+reg.params[1]*theta,color='r')
plt.xlabel('SPX return')
plt.ylabel('VIX return')
plt.title('SPX & VIX')

plt.subplot(2,2,2)
plt.scatter(selloff.SPX,selloff.VIX_lag1)
plt.xlabel('SPX return')
plt.ylabel('VIX lag1 return')
plt.title('SPX & VIX_lag1')

plt.subplot(2,2,3)
plt.scatter(selloff.SPX,selloff.VIX_lag2)
plt.plot(theta,reg_lag2.params[0]+reg_lag2.params[1]*theta,color='r')
plt.xlabel('SPX return')
plt.ylabel('VIX lag2 return')
plt.title('SPX & VIX_lag2')

plt.subplot(2,2,4)
plt.scatter(selloff.SPX,selloff.VIX_lag3)
plt.plot(theta,reg_lag3.params[0]+reg_lag3.params[1]*theta,color='r')
plt.xlabel('SPX return')
plt.ylabel('VIX lag3 return')
plt.title('SPX & VIX_lag3')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# =============================================================================
#  VIX and VIX futures
# =============================================================================

futures = data[data['m_localSymbol']=='VIX'][['date','logRet_daily']]
futures.rename(columns={'logRet_daily':'VIX'},inplace=True)
futures = futures[futures['date'].isin(selloff['date'])] # select days of selloff
futures['const']=1

temp=data.dropna(axis=0)
temp=temp[temp['date']<temp['expiry']]  # eliminate data on expire date, no price change
temp.rename(columns={'logRet_daily':'fRet'},inplace=True)
temp['t']=(temp['expiry']-temp['date'])/dt.timedelta(days=365)
temp[temp['date'].isin(selloff['date'])]

# merge no lag to futures
futures=futures.merge(temp[['date','symbol','t','fRet']],on='date',how='inner')

# merge lag 30 mins to futures
temp_lag1=temp.copy()
temp_lag1.rename(columns={'date':'date_lag1'},inplace=True)
temp_lag1['date']=temp_lag1['date_lag1']-dt.timedelta(hours=0.5)
temp_lag1.rename(columns={'fRet':'fRet_lag1'},inplace=True)
temp_lag1['t_lag1']=(temp_lag1['expiry']-temp_lag1['date_lag1'])/dt.timedelta(days=365)
futures=futures.merge(temp_lag1[['date','symbol','t_lag1','fRet_lag1']],on=['date','symbol'],how='left')

# merge lag 60 mins to futures
temp_lag2=temp.copy()
temp_lag2.rename(columns={'date':'date_lag2'},inplace=True)
temp_lag2['date']=temp_lag2['date_lag2']-dt.timedelta(hours=1)
temp_lag2.rename(columns={'fRet':'fRet_lag2'},inplace=True)
temp_lag2['t_lag2']=(temp_lag2['expiry']-temp_lag2['date_lag2'])/dt.timedelta(days=365)
futures=futures.merge(temp_lag2[['date','symbol','t_lag2','fRet_lag2']],on=['date','symbol'],how='left')

# merge lag 60 mins to futures
temp_lag3=temp.copy()
temp_lag3.rename(columns={'date':'date_lag3'},inplace=True)
temp_lag3['date']=temp_lag3['date_lag3']-dt.timedelta(hours=1.5)
temp_lag3.rename(columns={'fRet':'fRet_lag3'},inplace=True)
temp_lag3['t_lag3']=(temp_lag3['expiry']-temp_lag3['date_lag3'])/dt.timedelta(days=365)
futures=futures.merge(temp_lag3[['date','symbol','t_lag3','fRet_lag3']],on=['date','symbol'],how='left')

# futures.isna().sum()

VIX_fRet_corr = np.corrcoef([futures.dropna()['VIX'],futures.dropna()['fRet'],
                             futures.dropna()['fRet_lag1'],futures.dropna()['fRet_lag2'],
                             futures.dropna()['fRet_lag3']])
VIX_fRet_corr_table = pd.DataFrame(data=VIX_fRet_corr,index=['VIX','fRet','fRet_lag1','fRet_lag2','fRet_lag3'],
                                   columns=['VIX','fRet','fRet_lag1','fRet_lag2','fRet_lag3'])

# scatter plots for futures and VIX
plt.figure(figsize=(8,6))
plt.subplot(2,2,1)
plt.scatter(futures.VIX,futures.fRet)
plt.xlabel('VIX')
plt.ylabel('VIX future')
plt.title('VIX & VIX future')

plt.subplot(2,2,2)
plt.scatter(futures.VIX,futures.fRet_lag1)
plt.xlabel('VIX')
plt.ylabel('lag1 VIX future')
plt.title('VIX & lag1 VIX future')

plt.subplot(2,2,3)
plt.scatter(futures.VIX,futures.fRet_lag2)
plt.xlabel('VIX')
plt.ylabel('lag2 VIX future')
plt.title('VIX & lag2 VIX future')

plt.subplot(2,2,4)
plt.scatter(futures.VIX,futures.fRet_lag3)
plt.xlabel('VIX')
plt.ylabel('lag3 VIX future')
plt.title('VIX & lag3 VIX future')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# scatter plots for futures and t
plt.figure(figsize=(8,6))
plt.subplot(2,2,1)
plt.scatter(futures.t,futures.fRet)
plt.xlabel('t')
plt.ylabel('VIX future')
plt.title('t & VIX future')

plt.subplot(2,2,2)
plt.scatter(futures.t_lag1,futures.fRet_lag1)
plt.xlabel('t_lag1')
plt.ylabel('lag1 VIX future')
plt.title('t_lag1 & lag1 VIX future')

plt.subplot(2,2,3)
plt.scatter(futures.t_lag2,futures.fRet_lag2)
plt.xlabel('t_lag2')
plt.ylabel('lag2 VIX future')
plt.title('t_lag2 & lag2 VIX future')

plt.subplot(2,2,4)
plt.scatter(futures.t_lag3,futures.fRet_lag3)
plt.xlabel('t_lag3')
plt.ylabel('lag3 VIX future')
plt.title('t_lag3 & lag3 VIX future')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# no lag 
fx=futures[['const','VIX','t']].to_numpy()
fy=futures.fRet.to_numpy()
freg = sm.OLS(fy, fx).fit()
print(freg.summary())

fxt2=fx.copy()
fxt2[:,2]=fxt2[:,2]**2  # try t square
fregt2 = sm.OLS(fy, fxt2).fit()
print(fregt2.summary())

# lag 30 mins
fx_lag1=futures.dropna(subset=['fRet_lag1'])[['const','VIX','t']].to_numpy()
fy_lag1=futures.dropna(subset=['fRet_lag1']).fRet_lag1.to_numpy()
freg_lag1 = sm.OLS(fy_lag1, fx_lag1).fit()
print(freg_lag1.summary())

# lag 60 mins
fx_lag2=futures.dropna()[['const','VIX','t']].to_numpy()
fy_lag2=futures.dropna().fRet_lag2.to_numpy()
freg_lag2 = sm.OLS(fy_lag2, fx_lag2).fit()
print(freg_lag2.summary())

# lag 90 mins
fx_lag3=futures.dropna()[['const','VIX','t']].to_numpy()
fy_lag3=futures.dropna().fRet_lag3.to_numpy()
freg_lag3 = sm.OLS(fy_lag3, fx_lag3).fit()
print(freg_lag3.summary())

 