# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 21:23:24 2018

@author: jhshim
"""

#%%
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt

plt.close("all")

# FE557 상품거래기법 과제

sm = 120 # 베타추정을 위한 회귀분석기간, 변경가능
market = 'Energy' # Energy or Gold


#%% data loading

if market == 'Energy':
    df = pd.read_excel(r'C:\Users\user\Desktop\dataset.xlsx',sheetname='Energy_return')
    df = df[['NRE','energy','SnP','risk free scaled']].rename(columns = {'NRE':'NRE','energy':'CI','SnP':'Market','risk free scaled':'rf'}).reset_index()
elif market == 'Gold':
    df = pd.read_excel(r'C:\Users\user\Desktop\dataset.xlsx',sheetname='GOLD_return')
    df = df[['NRE','GOLD_CI','SnP','risk free scaled']].rename(columns = {'NRE':'NRE','GOLD_CI':'CI','SnP':'Market','risk free scaled':'rf'}).reset_index()

df['YM'] = df['index'].map(lambda x: 100*x.year + x.month)

y = df['NRE']
CI= df['CI']
SnP= df['Market']

#%% rebalance using betas from regressions
print('')
print('-'*40)
print('Rebalanced pf using betas from regressions')
print('sampling month: {}, backtested month: {}'.format(sm,len(df)-sm-1))
print('')

coef=[]
t_val=[]
f_val=[]
r_sqr=[]

for i in np.arange(0,len(df)-sm-1):
    
    y = df['NRE'].iloc[i:i+sm]
    x1= df['CI'].iloc[i:i+sm]
    x2= df['Market'].iloc[i:i+sm]
    x = pd.concat([y, x1, x2],axis=1)
    models = ols(formula='y ~ x1 + x2',data = x).fit()
    coef.append(models.params)
    t_val.append(models.tvalues)
    f_val.append(models.fvalue)
    r_sqr.append(models.rsquared)
    
    
coef_df=pd.DataFrame(coef)
t_val_df=pd.DataFrame(t_val)
t_val_df=t_val_df[['Intercept', 'x1', 'x2']].rename(columns={'Intercept':'I_t','x1':'x1_t','x2':'x2_t'})
f_val_df=pd.DataFrame(f_val).rename(columns = {0:'f_val'})
r_sqr_df=pd.DataFrame(r_sqr).rename(columns = {0:'r_sqr'})

coef_df=pd.DataFrame(coef)
coef_df['rf'] = 1-coef_df['x1']-coef_df['x2']
df1 = df[sm:-1].reset_index(drop=True)   

sample = pd.DataFrame()
sample['mimic'] = df1['CI'].mul(coef_df.x1,axis=0)+df1['Market'].mul(coef_df.x2,axis=0)+df1['rf'].mul(coef_df.rf,axis=0)
sample['nre'] = df1['NRE']
sample['diff'] = sample['mimic']-sample['nre']
sample['index'] = df1['index']  
sample = sample.set_index('index')

result = sample.describe(include='all')
result = result.append(pd.Series(sample.skew(),name='skewness'))
result = result.append(pd.Series(sample.kurtosis(),name='kurtosis'))
print(result)

print('Sharpe Ratio of NRE          : %.3f' % ((sample.nre.values-df['rf'].loc[sm+1::1]).mean()*12/((sample.nre.values-df['rf'].loc[sm+1::1]).std()*np.sqrt(12))))
print('Sharpe Ratio of Replicated PF: %.3f' % ((sample.mimic.values-df['rf'].loc[sm+1::1]).mean()*12/((sample.mimic.values-df['rf'].loc[sm+1::1]).std()*np.sqrt(12))))

print('')



#%% figures

print('-'*40)
plt.figure(0) #figure 2 of PIMCO
coef_df.index = sample.index
plt.plot(coef_df['x1'],label='CI')
plt.plot(coef_df['x2'],label='S&P')
plt.legend()
plt.title('Replicating {1} NRE: rolling betas using {0} monthly observations'.format(sm,market))   


plt.figure(1) #figure 6 Overall cumulative returns
cp_sample = (sample[['mimic','nre']]+1).cumprod().reset_index()
cp_sample.loc[-1] = [(sample.index[0]-pd.Timedelta(1,'M')).replace(hour=0, minute=0, second=0), 1, 1]
cp_sample = cp_sample.sort_index()
cp_sample = cp_sample.set_index(cp_sample['index']).drop(['index'],axis=1)
cp_sample['Excess Return'] = cp_sample.mimic-cp_sample.nre
plt.plot(cp_sample)
plt.legend(['Relicated PF','NRE','Excess Return'])
plt.ylabel('Growth of a dollor($)')
plt.title('Growth of a dollar for replicating PF and NRE')
