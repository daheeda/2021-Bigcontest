#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import keras
import pandas as pd
from keras.layers import Dense, Activation, LSTM,Dropout,Bidirectional
from sklearn.preprocessing import MinMaxScaler
import keras.backend as K
from statsmodels.tsa.arima_model import ARIMA
import itertools
import warnings
warnings.filterwarnings(action='ignore')


# In[80]:


varday = pd.DataFrame(index=range(0,41),
                     columns=['행정동명','var_일별_7월','var_일별_8월'])
varmonth = pd.DataFrame(index=range(0,41),
                     columns=['행정동명','var_월별_7월','var_월별_8월'])
lstmdf = pd.DataFrame(index=range(0,41),
                     columns=['행정동명','lstm_7월','lstm_8월'])
arimaday = pd.DataFrame(index=range(0,41),
                     columns=['행정동명','arima_일별_7월','arima_일별_8월'])
arimamonth = pd.DataFrame(index=range(0,41),
                     columns=['행정동명','arima_월별_7월','arima_월별_8월'])


# In[81]:


emdnm=pd.read_csv('전체테이블.csv',encoding='euc-kr')['emd_nm'].unique() #행정동명 추출


# ## Data HEAD

# In[33]:


print(pd.read_csv('전체테이블.csv',encoding='euc-kr').head(1))


# In[34]:


print(pd.read_csv('알수없음테이블.csv',encoding='euc-kr').head(1))


# # 1. VAR - day

# In[13]:


class var_day:  # 행정동 나누고, lag 변수 생성, trend, seasonal 추가
    def __init__(self, df, emdnm,lag):
        self.df=df
        self.emdnm=emdnm
        self.lag=lag
    def varset(self):    
        self.df['base_date']=pd.to_datetime(self.df['base_date'])
        self.df.index=self.df['base_date']
        self.df.set_index('base_date', inplace=True)
        self.df=self.df.drop(['city','loc.1','loc.2','temp.2','prec.2','hum.2','snow.2','area_num',
                              '거리두기단계','집합금지','temp.1','prec.1','hum.1','snow.1'],axis=1)
        self.df = self.df[self.df['emd_nm']==self.emdnm]
        del self.df['emd_nm']
        self.df.asfreq('D')[self.df.asfreq('D').isnull().sum(axis=1)>0]
        self.df = self.df.asfreq('D',method='ffill')
        for s in range(1, self.lag+1): 
            self.df['shift_{}'.format(s)] = self.df['g_sum'].shift(s)
            self.df['shift_{}'.format(s)].fillna(method='bfill',inplace=True)
        result = sm.tsa.seasonal_decompose(self.df['g_sum'],model='additive',period=12)
        data_trend = pd.DataFrame(result.trend)
        data_trend.fillna(method='ffill',inplace=True)
        data_trend.fillna(method='bfill',inplace=True)
        data_seasonal = pd.DataFrame(result.seasonal)
        data_seasonal.fillna(method='ffill',inplace=True)
        data_seasonal.fillna(method='bfill',inplace=True)
        adddf = pd.concat([self.df,data_trend,data_seasonal],axis=1)
        return adddf


# In[88]:


class backward_diff: #backward 선택된 변수 & 차분 2번 나머지는 drop & var
    def __init__(self, df,load,emdnm):
        self.df=df
        self.load=load
        self.emdnm=emdnm
        
    def regvar(self):
        df=self.df
        data_y=df['g_sum']
        data_x=df.drop(df[['g_sum','base_year']],axis=1)
        model_reg1 = sm.OLS(data_y,data_x).fit()
        varlist = data_x.columns.tolist()
        y = data_y
        sv_per_step = []
        ad_r = []
        steps = []
        step = 0
        while len(varlist) > 0 :
            X = sm.add_constant(data_x[varlist])
            p_values = sm.OLS(y,X).fit().pvalues[1:]
            max_pval = p_values.max()
            if max_pval >= 0.05 :
                remove_var = p_values.idxmax()
                varlist.remove(remove_var)
                step +=1
                steps.append(step)
                adj_r = sm.OLS(y,sm.add_constant(data_x[varlist]))
                ad_r.append(adj_r)
                sv_per_step.append(varlist.copy())
            else :
                break
                
        data_select=data_y        
        for i in range(0,len(varlist)):
            data_x_select = data_x[[varlist[i]]]
            data_select = pd.concat([data_x_select,data_select],axis=1)
        
        varlist.append('g_sum')
        for i in range (0,len(varlist)) :
            adfuller_test = adfuller(data_select[[varlist[i]]], autolag= "AIC")
            if adfuller_test[1] >=0.05 :
                data_select[[varlist[i]]] = data_select[[varlist[i]]].diff().dropna()  # 1차 차분
                data_select.fillna(method='bfill',inplace=True)
        
        for i in range (0,len(varlist)) :
            adfuller_test = adfuller(data_select[[varlist[i]]], autolag= "AIC")
            if adfuller_test[1] >=0.05 :
                data_select[[varlist[i]]] = data_select[[varlist[i]]].diff().dropna()  # 2차 차분
                data_select.fillna(method='bfill',inplace=True)
        
        data_select2=data_select
        arr=[]
        for i in range (0,len(varlist)) :
            adfuller_test = adfuller(data_select[[varlist[i]]], autolag= "AIC")
            if adfuller_test[1] >=0.05 :
                data_select2=data_select2.drop(data_select2[[varlist[i]]],axis=1) #drop
            else :
                arr.append(i)
        
        datanew=np.asarray(data_select2)
        
        forecasting_model = VAR(datanew) #var
        results = forecasting_model.fit(len(arr)+1)
        lagg = data_select2.values[-len(arr)-1:]
        
        forecast = pd.DataFrame(results.forecast(y=lagg,steps=62),index=pd.date_range('2021-07-01',periods=62))
    
        df1gsum=var_day(self.load,self.emdnm,0).varset()
        forecast['g_sum'] = df1gsum['g_sum'].iloc[-62-1] + forecast.iloc[:,[-1]].cumsum()
        
        return forecast['g_sum']


# In[89]:


class backward_diff_2: # gsum차분 안할 때
    def __init__(self, df,load,emdnm):
        self.df=df
        self.load=load
        self.emdnm=emdnm
        
    def regvar(self):
        df=self.df
        data_y=df['g_sum']
        data_x=df.drop(df[['g_sum','base_year']],axis=1)
        model_reg1 = sm.OLS(data_y,data_x).fit()
        varlist = data_x.columns.tolist()
        y = data_y
        sv_per_step = []
        ad_r = []
        steps = []
        step = 0
        while len(varlist) > 0 :
            X = sm.add_constant(data_x[varlist])
            p_values = sm.OLS(y,X).fit().pvalues[1:]
            max_pval = p_values.max()
            if max_pval >= 0.05 :
                remove_var = p_values.idxmax()
                varlist.remove(remove_var)
                step +=1
                steps.append(step)
                adj_r = sm.OLS(y,sm.add_constant(data_x[varlist]))
                ad_r.append(adj_r)
                sv_per_step.append(varlist.copy())
            else :
                break
                
        data_select=data_y        
        for i in range(0,len(varlist)):
            data_x_select = data_x[[varlist[i]]]
            data_select = pd.concat([data_x_select,data_select],axis=1)
        
        varlist.append('g_sum')
        for i in range (0,len(varlist)) :
            adfuller_test = adfuller(data_select[[varlist[i]]], autolag= "AIC")
            if adfuller_test[1] >=0.05 :
                data_select[[varlist[i]]] = data_select[[varlist[i]]].diff().dropna()  # 1차 차분
                data_select.fillna(method='bfill',inplace=True)
        
        for i in range (0,len(varlist)) :
            adfuller_test = adfuller(data_select[[varlist[i]]], autolag= "AIC")
            if adfuller_test[1] >=0.05 :
                data_select[[varlist[i]]] = data_select[[varlist[i]]].diff().dropna()  # 2차 차분
                data_select.fillna(method='bfill',inplace=True)
        
        data_select2=data_select
        arr=[]
        for i in range (0,len(varlist)) :
            adfuller_test = adfuller(data_select[[varlist[i]]], autolag= "AIC")
            if adfuller_test[1] >=0.05 :
                data_select2=data_select2.drop(data_select2[[varlist[i]]],axis=1) #drop
            else :
                arr.append(i)
        
        datanew=np.asarray(data_select2)
        
        forecasting_model = VAR(datanew) #var
        results = forecasting_model.fit(len(arr)+1)
        lagg = data_select2.values[-len(arr)-1:]
        
        forecast = pd.DataFrame(results.forecast(y=lagg,steps=62),index=pd.date_range('2021-07-01',periods=62))
    
        df1gsum=var_day(self.load,self.emdnm,0).varset()
        forecast['g_sum']=forecast.iloc[:,[-1]].cumsum()
        
        return forecast['g_sum']


# In[163]:


for i in range(0, len(emdnm)):
    if emdnm[i]=='효돈동':
        value=backward_diff(var_day(pd.read_csv('전체테이블.csv',encoding='euc-kr'),emdnm[i],1).varset(),pd.read_csv('전체테이블.csv',encoding='euc-kr'),emdnm[i]).regvar()
        varday.iloc[i,0]=emdnm[i]
        varday.iloc[i,1]=value[:31].sum()
        varday.iloc[i,2]=value[31:].sum()
    elif (emdnm[i]=='서홍동')|(emdnm[i]=='송산동')|(emdnm[i]=='정방동')|(emdnm[i]=='중앙동')|(emdnm[i]=='천지동')|(emdnm[i]=='노형동')|(emdnm[i]=='도두동')|(emdnm[i]=='삼양동')|(emdnm[i]=='아라동')|(emdnm[i]=='연동')|(emdnm[i]=='오라동')|(emdnm[i]=='이도2동')|(emdnm[i]=='화북동')|(emdnm[i]=='한림읍'):
        value=backward_diff_2(var_day(pd.read_csv('전체테이블.csv',encoding='euc-kr'),emdnm[i],10).varset(),pd.read_csv('전체테이블.csv',encoding='euc-kr'),emdnm[i]).regvar()
        varday.iloc[i,0]=emdnm[i]
        varday.iloc[i,1]=value[:31].sum()
        varday.iloc[i,2]=value[31:].sum()
    else :
        value=backward_diff(var_day(pd.read_csv('전체테이블.csv',encoding='euc-kr'),emdnm[i],10).varset(),pd.read_csv('전체테이블.csv',encoding='euc-kr'),emdnm[i]).regvar()
        varday.iloc[i,0]=emdnm[i]
        varday.iloc[i,1]=value[:31].sum()
        varday.iloc[i,2]=value[31:].sum()


# # 2. VAR - month

# In[137]:


class var_month:
    def __init__(self, df, emdnm,lag):
        self.df=df
        self.emdnm=emdnm
        self.lag=lag
        
    def varset_month(self):    
        self.df['base_date']=pd.to_datetime(self.df['base_date'])
        self.df['date']=self.df['base_date'].apply(lambda x: x.strftime('%Y%m'))
        self.df=self.df.drop(['city','loc.1','loc.2','temp.2','prec.2','hum.2','snow.2','area_num',
                              '거리두기단계','집합금지','temp.1','prec.1','hum.1','snow.1','base_year','base_month'],axis=1)
        self.df = self.df[self.df['emd_nm']==self.emdnm]
        del self.df['emd_nm']
        
        self.df=self.df.groupby(['date']).sum()
        self.df['Date']=self.df.index
        self.df['Date']=pd.to_datetime(self.df['Date'], format='%Y%m').dt.strftime('%Y-%m')
        self.df.set_index('Date',inplace=True)
        
        for s in range(1, self.lag+1): 
            self.df['shift_{}'.format(s)] = self.df['g_sum'].shift(s)
            self.df['shift_{}'.format(s)].fillna(method='bfill',inplace=True)
        
        result = sm.tsa.seasonal_decompose(self.df['g_sum'],model='additive',period=12)
        data_trend = pd.DataFrame(result.trend)
        data_trend.fillna(method='ffill',inplace=True)
        data_trend.fillna(method='bfill',inplace=True)
        data_seasonal = pd.DataFrame(result.seasonal)
        data_seasonal.fillna(method='ffill',inplace=True)
        data_seasonal.fillna(method='bfill',inplace=True)
        adddf = pd.concat([self.df,data_trend,data_seasonal],axis=1)
        
        return adddf


# In[138]:


class var_month_2:
    def __init__(self, df, emdnm,lag):
        self.df=df
        self.emdnm=emdnm
        self.lag=lag
        
    def varset_month(self):    
        self.df['base_date']=pd.to_datetime(self.df['base_date'])
        self.df['date']=self.df['base_date'].apply(lambda x: x.strftime('%Y%m'))
        self.df=self.df.drop(['city','loc.1','loc.2','temp.2','prec.2','hum.2','snow.2','area_num',
                              '거리두기단계','집합금지','temp.1','prec.1','hum.1','snow.1','base_year','base_month'],axis=1)
        self.df = self.df[self.df['emd_nm']==self.emdnm]
        del self.df['emd_nm']
        
        self.df=self.df.groupby(['date']).sum()
        self.df['Date']=self.df.index
        self.df['Date']=pd.to_datetime(self.df['Date'], format='%Y%m').dt.strftime('%Y-%m')
        self.df.set_index('Date',inplace=True)
        
        for s in range(1, self.lag+1): 
            self.df['shift_{}'.format(s)] = self.df['g_sum'].shift(s)
            self.df['shift_{}'.format(s)].fillna(method='bfill',inplace=True)
        
        return self.df


# In[139]:


class backward_diff_month: #backward 선택된 변수 & 차분 2번 나머지는 drop & var
    def __init__(self, df,load,emdnm):
        self.df=df
        self.load=load
        self.emdnm=emdnm
        
    def regvar(self):
        df=self.df
        data_y=df['g_sum']
        data_x=df.drop(df[['g_sum']],axis=1)
        model_reg1 = sm.OLS(data_y,data_x).fit()
        varlist = data_x.columns.tolist()
        y = data_y
        sv_per_step = []
        ad_r = []
        steps = []
        step = 0
        while len(varlist) > 0 :
            X = sm.add_constant(data_x[varlist])
            p_values = sm.OLS(y,X).fit().pvalues[1:]
            max_pval = p_values.max()
            if max_pval >= 0.05 :
                remove_var = p_values.idxmax()
                varlist.remove(remove_var)
                step +=1
                steps.append(step)
                adj_r = sm.OLS(y,sm.add_constant(data_x[varlist]))
                ad_r.append(adj_r)
                sv_per_step.append(varlist.copy())
            else :
                break
                
        data_select=data_y        
        for i in range(0,len(varlist)):
            data_x_select = data_x[[varlist[i]]]
            data_select = pd.concat([data_x_select,data_select],axis=1)
        
        varlist.append('g_sum')
        for i in range (0,len(varlist)) :
            adfuller_test = adfuller(data_select[[varlist[i]]], autolag= "AIC")
            if adfuller_test[1] >=0.05 :
                data_select[[varlist[i]]] = data_select[[varlist[i]]].diff().dropna()  # 1차 차분
                data_select.fillna(method='bfill',inplace=True)
        
        for i in range (0,len(varlist)) :
            adfuller_test = adfuller(data_select[[varlist[i]]], autolag= "AIC")
            if adfuller_test[1] >=0.05 :
                data_select[[varlist[i]]] = data_select[[varlist[i]]].diff().dropna()  # 2차 차분
                data_select.fillna(method='bfill',inplace=True)
        
        data_select2=data_select
        arr=[]
        for i in range (0,len(varlist)) :
            adfuller_test = adfuller(data_select[[varlist[i]]], autolag= "AIC")
            if adfuller_test[1] >=0.05 :
                data_select2=data_select2.drop(data_select2[[varlist[i]]],axis=1) #drop
            else :
                arr.append(i)
        
        datanew=np.asarray(data_select2)
        
        forecasting_model = VAR(datanew) #var
        results = forecasting_model.fit(2)
        lagg = data_select2.values[-2:]
        
        forecast = pd.DataFrame(results.forecast(y=lagg,steps=2),index=pd.date_range('2021-07',periods=2))
    
        df1gsum=var_month(self.load,self.emdnm,10).varset_month()
        forecast['g_sum'] = df1gsum['g_sum'].iloc[-3] + forecast.iloc[:,[-1]].cumsum()
        
        return forecast['g_sum']


# In[140]:


class backward_diff_month_2: # gsum 차분 안하는 경우
    def __init__(self, df,load,emdnm):
        self.df=df
        self.load=load
        self.emdnm=emdnm
        
    def regvar(self):
        df=self.df
        data_y=df['g_sum']
        data_x=df.drop(df[['g_sum']],axis=1)
        model_reg1 = sm.OLS(data_y,data_x).fit()
        varlist = data_x.columns.tolist()
        y = data_y
        sv_per_step = []
        ad_r = []
        steps = []
        step = 0
        while len(varlist) > 0 :
            X = sm.add_constant(data_x[varlist])
            p_values = sm.OLS(y,X).fit().pvalues[1:]
            max_pval = p_values.max()
            if max_pval >= 0.05 :
                remove_var = p_values.idxmax()
                varlist.remove(remove_var)
                step +=1
                steps.append(step)
                adj_r = sm.OLS(y,sm.add_constant(data_x[varlist]))
                ad_r.append(adj_r)
                sv_per_step.append(varlist.copy())
            else :
                break
                
        data_select=data_y        
        for i in range(0,len(varlist)):
            data_x_select = data_x[[varlist[i]]]
            data_select = pd.concat([data_x_select,data_select],axis=1)
        
        varlist.append('g_sum')
        for i in range (0,len(varlist)) :
            adfuller_test = adfuller(data_select[[varlist[i]]], autolag= "AIC")
            if adfuller_test[1] >=0.05 :
                data_select[[varlist[i]]] = data_select[[varlist[i]]].diff().dropna()  # 1차 차분
                data_select.fillna(method='bfill',inplace=True)
        
        for i in range (0,len(varlist)) :
            adfuller_test = adfuller(data_select[[varlist[i]]], autolag= "AIC")
            if adfuller_test[1] >=0.05 :
                data_select[[varlist[i]]] = data_select[[varlist[i]]].diff().dropna()  # 2차 차분
                data_select.fillna(method='bfill',inplace=True)
        
        data_select2=data_select
        arr=[]
        for i in range (0,len(varlist)) :
            adfuller_test = adfuller(data_select[[varlist[i]]], autolag= "AIC")
            if adfuller_test[1] >=0.05 :
                data_select2=data_select2.drop(data_select2[[varlist[i]]],axis=1) #drop
            else :
                arr.append(i)
        
        datanew=np.asarray(data_select2)
        
        forecasting_model = VAR(datanew) #var
        results = forecasting_model.fit(2)
        lagg = data_select2.values[-2:]
        
        forecast = pd.DataFrame(results.forecast(y=lagg,steps=2),index=pd.date_range('2021-07',periods=2))
    
        df1gsum=var_month(self.load,self.emdnm,10).varset_month()
        forecast['g_sum']=forecast.iloc[:,[-1]].cumsum()
        
        return forecast['g_sum']


# In[141]:


class backward_diff_time_x: #backward 선택된 변수 & 차분 2번 나머지는 drop & var
    def __init__(self, df,load,emdnm):
        self.df=df
        self.load=load
        self.emdnm=emdnm
        
    def regvar(self):
        df=self.df
        data_y=df['g_sum']
        data_x=df.drop(df[['g_sum']],axis=1)
        model_reg1 = sm.OLS(data_y,data_x).fit()
        varlist = data_x.columns.tolist()
        y = data_y
        sv_per_step = []
        ad_r = []
        steps = []
        step = 0
        while len(varlist) > 0 :
            X = sm.add_constant(data_x[varlist])
            p_values = sm.OLS(y,X).fit().pvalues[1:]
            max_pval = p_values.max()
            if max_pval >= 0.05 :
                remove_var = p_values.idxmax()
                varlist.remove(remove_var)
                step +=1
                steps.append(step)
                adj_r = sm.OLS(y,sm.add_constant(data_x[varlist]))
                ad_r.append(adj_r)
                sv_per_step.append(varlist.copy())
            else :
                break
                
        data_select=data_y        
        for i in range(0,len(varlist)):
            data_x_select = data_x[[varlist[i]]]
            data_select = pd.concat([data_x_select,data_select],axis=1)
        
        varlist.append('g_sum')
        for i in range (0,len(varlist)) :
            adfuller_test = adfuller(data_select[[varlist[i]]], autolag= "AIC")
            if adfuller_test[1] >=0.05 :
                data_select[[varlist[i]]] = data_select[[varlist[i]]].diff().dropna()  # 1차 차분
                data_select.fillna(method='bfill',inplace=True)
        
        for i in range (0,len(varlist)) :
            adfuller_test = adfuller(data_select[[varlist[i]]], autolag= "AIC")
            if adfuller_test[1] >=0.05 :
                data_select[[varlist[i]]] = data_select[[varlist[i]]].diff().dropna()  # 2차 차분
                data_select.fillna(method='bfill',inplace=True)
        
        data_select2=data_select
        arr=[]
        for i in range (0,len(varlist)) :
            adfuller_test = adfuller(data_select[[varlist[i]]], autolag= "AIC")
            if adfuller_test[1] >=0.05 :
                data_select2=data_select2.drop(data_select2[[varlist[i]]],axis=1) #drop
            else :
                arr.append(i)
        
        datanew=np.asarray(data_select2)
        
        forecasting_model = VAR(datanew) #var
        results = forecasting_model.fit(2)
        lagg = data_select2.values[-2:]
        
        forecast = pd.DataFrame(results.forecast(y=lagg,steps=2),index=pd.date_range('2021-07',periods=2))
    
        df1gsum=var_month_2(self.load,self.emdnm,10).varset_month()
        
        if self.emdnm=='한경면' :
            forecast['g_sum']=forecast.iloc[:,[-1]].cumsum()
        else :
            forecast['g_sum'] = df1gsum['g_sum'].iloc[-3] + forecast.iloc[:,[-1]].cumsum()
        
        return forecast['g_sum']


# In[164]:


for i in range(0, len(emdnm)):
    if (emdnm[i]=='구좌읍')|(emdnm[i]=='조천읍')|(emdnm[i]=='한경면')|(emdnm[i]=='한림읍'):
        value=backward_diff_time_x(var_month_2(pd.read_csv('전체테이블.csv',encoding='euc-kr'),'조천읍',5).varset_month(),pd.read_csv('전체테이블.csv',encoding='euc-kr'),'조천읍').regvar()
        varmonth.iloc[i,0]=emdnm[i]
        varmonth.iloc[i,1]=value[:1].sum()
        varmonth.iloc[i,2]=value[1:].sum()        
    elif (emdnm[i]=='서홍동')|(emdnm[i]=='송산동')|(emdnm[i]=='정방동')|(emdnm[i]=='천지동')|(emdnm[i]=='도두동')|(emdnm[i]=='아라동')|(emdnm[i]=='이호동'):
        value=backward_diff_month_2(var_month(pd.read_csv('전체테이블.csv',encoding='euc-kr'),emdnm[i],10).varset_month(),pd.read_csv('전체테이블.csv',encoding='euc-kr'),emdnm[i]).regvar()
        varmonth.iloc[i,0]=emdnm[i]
        varmonth.iloc[i,1]=value[:1].sum()
        varmonth.iloc[i,2]=value[1:].sum()        
    else :
        value=backward_diff_month(var_month(pd.read_csv('전체테이블.csv',encoding='euc-kr'),emdnm[i],10).varset_month(),pd.read_csv('전체테이블.csv',encoding='euc-kr'),emdnm[i]).regvar()
        varmonth.iloc[i,0]=emdnm[i]
        varmonth.iloc[i,1]=value[:1].sum()
        varmonth.iloc[i,2]=value[1:].sum()        


# # 3. LSTM

# In[102]:


class LSTM_run:
    def __init__(self, load, emdnm,target):
        self.load=load
        self.emdnm=emdnm
        self.target=target
    def run(self):
        
        df1=self.load[self.load['emd_nm']==self.emdnm]
        df1=df1[[self.target]] 
        
        sc = MinMaxScaler()
        df2=sc.fit_transform(df1)
        df1=pd.DataFrame(df2, columns=[self.target], index=df1.index)
        
        for i in range (0,62) :
    
            dfpred = df1.iloc[:-20-1,]  
            dfreal = df1.iloc[-20-1:,]

            for s in range(1, 21): # lag 추가
                dfpred['lag_{}'.format(s)] = dfpred[self.target].shift(s)
                dfreal['lag_{}'.format(s)] = dfreal[self.target].shift(s)

            #train, test set 나누기    
            x_train = dfpred.dropna().drop(self.target, axis=1)
            y_train = dfpred.dropna()[[self.target]]
            x_test = dfreal.dropna().drop(self.target, axis=1)
            y_test = dfreal.dropna()[[self.target]]
            x_trian=x_train.values
            y_train=y_train.values
            x_test=x_test.values
            y_test=y_test.values
    
            #학습위해 array 로 변경
            x_train=np.asarray(x_train)
            x_test=np.asarray(x_test)
            y_train=np.asarray(y_train)
            y_test=np.asarray(y_test) 
    
            x_train_t = x_train.reshape(x_train.shape[0],20,1)
            x_test_t = x_test.reshape(x_test.shape[0],20,1)

            # lstm 모델 학습
            K.clear_session()
            model = Sequential()
            model.add(LSTM( 10, input_shape=(20,1))) # timestep, feature
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer='adam')

            early_stop=EarlyStopping(monitor='loss', patience=7, verbose=1 )
            model.fit(x_train_t,y_train,epochs=20,batch_size=int(len(df1)/7),verbose=1,callbacks=[early_stop])
    
            #추가 된 데이터로 다시 1일씩 예측
            y_hat = model.predict(x_test_t)
            y_hat = { self.target : y_hat.sum() }
            df1=df1.append(y_hat, ignore_index=True)
        
        df1=sc.inverse_transform(df1)
        
        return df1[-62:]


# In[118]:


for i in range(0, len(emdnm)):
    value=LSTM_run(pd.read_csv('전체테이블.csv',encoding='euc-kr'),emdnm[0],'g_sum').run()
    lstmdf.iloc[i,0]=emdnm[i]
    lstmdf.iloc[i,1]=value[-62:-31].sum()
    lstmdf.iloc[i,2]=value[-31:].sum()


# # 4. ARIMA - day

# In[106]:


#행정동나누기
class setdata_arima: 
    def __init__(self, df, emdnm):
        self.df=df
        self.emdnm=emdnm
    def set_arima(self):    
        self.df['base_date']=pd.to_datetime(self.df['base_date'])
        self.df.index=self.df['base_date']
        self.df.set_index('base_date', inplace=True)
        self.df = self.df[self.df['emd_nm']==self.emdnm]
        self.df.asfreq('D')[self.df.asfreq('D').isnull().sum(axis=1)>0]
        self.df = self.df.asfreq('D',method='ffill')
        return self.df


# In[107]:


#차분
class gsum_diff: 
    def __init__(self, df,emd_nm):
        self.df=df
        self.emd_nm=emd_nm
        
    def difff(self):
        df=self.df
        adfuller_test=adfuller(self.df[['g_sum']],autolag='AIC')
        if adfuller_test[1]>=0.05 :
            self.df['g_sum_diff(1)']=self.df[['g_sum']].diff().dropna()
            self.df.fillna(method='bfill',inplace=True)
        else:
            self.df['g_sum_diff(1)']=self.df[['g_sum']]
        
        adfuller_test2=adfuller(self.df[['g_sum_diff(1)']],autolag='AIC')
        if adfuller_test2[1]>=0.05 :
            self.df['g_sum_diff(2)']=self.df[['g_sum_diff(1)']].diff().dropna()
            self.df.fillna(method='bfill',inplace=True)
        else:
            self.df['g_sum_diff(2)']=self.df[['g_sum_diff(1)']]
            
        return self.df


# In[110]:


#ARIMA
class arima: 
    def __init__(self, df,emdnm):
        self.df=df
        self.emdnm=emdnm
    
    def arima_run(self):
        p=list(range(0,3))
        d=[0,1,2]
        q=list(range(0,3))
        
        ss=list(itertools.product(p,d,q))

        aic_set=[]
        pr_set=[]

        for i in ss:
            try:
                model1=ARIMA(self.df['g_sum_diff(2)'].values,order=i)
                result=model1.fit()
                aic_set.append(result.aic)
                pr_set.append(i)
    
            except:
                continue


        min_aic=min(aic_set)
        best_pr=pr_set[aic_set.index(min_aic)]

        md1=ARIMA(self.df['g_sum_diff(2)'].values,order=best_pr)
        md1_result=md1.fit()
        pred=md1_result.forecast(steps=62)
        pred1=pd.DataFrame(pred[0])
        
        if all(self.df['g_sum']==self.df['g_sum_diff(2)'])==True :
            
            pred1['g_sum_forecast'] = pred1
            
        else :
            
            pred1['g_sum_forecast'] = self.df['g_sum'].iloc[-62-1] + pred1.cumsum()
            pred1.index=pd.date_range('2021-07-01',periods=62)
        
        return pred1


# In[111]:


for i in range(0, len(emdnm)):
    value=arima(gsum_diff(setdata_arima(pd.read_csv('전체테이블.csv',encoding='euc-kr').iloc[:,[0,3,8]],emdnm[i]).set_arima(),emdnm[i]).difff(),emdnm[i]).arima_run()
    arimaday.iloc[i,0]=emdnm[i]
    arimaday.iloc[i,1]=value.iloc[:31,1].sum()
    arimaday.iloc[i,2]=value.iloc[31:,1].sum()


# # 5. ARIMA - month

# In[114]:


#행정동/월별로나누기
class arima_month: 
    def __init__(self, df, emdnm):
        self.df=df
        self.emdnm=emdnm
    def set_arima(self):    
        self.df['base_date']=pd.to_datetime(self.df['base_date'])
        self.df['date']=self.df['base_date'].apply(lambda x: x.strftime('%Y%m'))
        self.df.index=self.df['base_date']
        self.df.set_index('base_date', inplace=True)
        self.df = self.df[self.df['emd_nm']==self.emdnm]
        del self.df['emd_nm']
        
        self.df=self.df.groupby(['date']).sum()
        self.df['Date']=self.df.index
        self.df['Date']=pd.to_datetime(self.df['Date'], format='%Y%m').dt.strftime('%Y-%m')
        self.df.set_index('Date',inplace=True)
        
        return self.df


# In[115]:


#ARIMA
class arima2: 
    def __init__(self, df,emdnm):
        self.df=df
        self.emdnm=emdnm
    
    def arima_run(self):
        p=list(range(0,3))
        d=[0,1,2]
        q=list(range(0,3))
        
        ss=list(itertools.product(p,d,q))

        aic_set=[]
        pr_set=[]

        for i in ss:
            try:
                model1=ARIMA(self.df['g_sum_diff(2)'].values,order=i)
                result=model1.fit()
                aic_set.append(result.aic)
                pr_set.append(i)
    
            except:
                continue


        min_aic=min(aic_set)
        best_pr=pr_set[aic_set.index(min_aic)]

        md1=ARIMA(self.df['g_sum_diff(2)'].values,order=best_pr)
        md1_result=md1.fit()
        pred=md1_result.forecast(steps=2)
        pred1=pd.DataFrame(pred[0])
        
        if all(self.df['g_sum']==self.df['g_sum_diff(2)'])==True :
            
            pred1['g_sum_forecast'] = pred1
            
        else :
            
            pred1['g_sum_forecast'] = self.df['g_sum'].iloc[-2-1] + pred1.cumsum()
            pred1.index=['2021-07','2021-08']
        
        pd.options.display.float_format = '{:.5f}'.format
        
        return pred1


# In[156]:


for i in range(0, len(emdnm)):
    value=arima2(gsum_diff(arima_month(pd.read_csv('전체테이블.csv',encoding='euc-kr').iloc[:,[0,3,8]],emdnm[i]).set_arima(),emdnm[i]).difff(),emdnm[i]).arima_run()
    arimamonth.iloc[i,0]=emdnm[i]
    arimamonth.iloc[i,1]=value.iloc[0,1].sum()
    arimamonth.iloc[i,2]=value.iloc[1:,1].sum()


# # 6. 알수없음 - LSTM

# In[120]:


nondf=pd.read_csv('알수없음테이블.csv',encoding='euc-kr').groupby('base_date')['em_g'].agg(**{'em_g':'sum'}).reset_index()
nondf['emd_nm']='알수없음'


# In[121]:


알수없음=LSTM_run(nondf,'알수없음','em_g').run()


# # 최종예측결과
# 이상치 제거 후 조화평균을 통한 배출량 예측

# In[147]:


알수없음 = {'행정동명':'알수없음','lstm_7월' :알수없음[-62:-31].sum(),'lstm_8월' :알수없음[-31:].sum()}


# In[180]:


def outlier(data,num):  #outlier 함수
    dt=pd.DataFrame()
    data2=data[num]
    quan25=np.percentile(data2.values,25)
    quan75=np.percentile(data2.values,75)
    iqr=(quan75-quan25)*1.5
    non_outlier=data2[(data2>quan25-iqr)&(data2<quan75)]
    return len(non_outlier)/(sum(1/non_outlier))


# In[181]:


varpred=pd.merge(varday,varmonth,on='행정동명')
arimapred=pd.merge(arimaday,arimamonth,on='행정동명')
vararipred=pd.merge(varpred,arimapred,on='행정동명')
finalpred=pd.merge(vararipred,lstmdf,on='행정동명')


# In[182]:


final_7=finalpred.loc[:,['var_일별_7월','var_월별_7월','lstm_7월','arima_일별_7월','arima_월별_7월']]
final_8=finalpred.loc[:,['var_일별_8월','var_월별_8월','lstm_8월','arima_일별_8월','arima_월별_8월']]
final_7_t=final_7.T
final_8_t=final_8.T


# In[183]:


final=pd.DataFrame(columns=['har_7','har_8'])
for i in range(0,41):
    final.loc[i,'har_7']=outlier(final_7_t,i)
for i in range(0,41):
    final.loc[i,'har_8']=outlier(final_8_t,i)

final['행정동명']=finalpred.iloc[:,0]
final=final.reindex(columns=['행정동명','har_7','har_8'])
final=final.append(알수없음, ignore_index=True)
final


# In[184]:


final.to_csv('제주도음식물쓰레기예측_최종.csv',header=True,index=False,encoding='euc-kr')

