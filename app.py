from datetime import timedelta, date
import datetime
from enum import auto
from flask import Flask, render_template, redirect, request
from flask.helpers import flash
import pandas as pd
import numpy as np
from sklearn.svm import SVR
import joblib

import plotly.graph_objs as go
import plotly.offline as py
import warnings
warnings.filterwarnings('ignore')
from flask_socketio import SocketIO
from tempfile import TemporaryDirectory

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

hist_df = pd.read_csv('Dataset_idrr.csv')
start_date = date(2022, 11, 30)
date_now = datetime.date.today()
date_tomorrow = date_now + datetime.timedelta(days = 1) 
end_date = date_now and date_tomorrow
df = pd.DataFrame()
for single_date in daterange(start_date, end_date):
    dfs = pd.read_html(f'https://www.xe.com/currencytables/?from=EUR&date={single_date.strftime("%Y-%m-%d")}')[0]
    dfs['Date'] = single_date.strftime("%Y-%m-%d")
    df = df.append(dfs)
df.to_csv('eur_data.csv')
idr_df = df[df['Currency'] == 'IDR']
idr_df = pd.concat([hist_df, idr_df], ignore_index=True)
idr_df.to_csv('idr.csv')     

length = len(idr_df)
data_day1=idr_df[length-1:]
data_day2=idr_df[length-2:length-1]
data_day7=idr_df[length-7:length-6]
data_day14=idr_df[length-14:length-13]
data_day31=idr_df[length-31:length-30]

change_1= float(data_day2['Units per EUR'])-float(data_day1['Units per EUR'])
change_7=float(data_day7['Units per EUR'])-float(data_day1['Units per EUR'])
change_14=float(data_day14['Units per EUR'])-float(data_day1['Units per EUR'])
change_31=float(data_day31['Units per EUR'])-float(data_day1['Units per EUR'])

price_day1=float(data_day1['Units per EUR'])

import os
app = Flask("_name_")
app.config["IMAGE_UPLOADS"] = "static/img/"
app.config["Graph_UPLOADS"] = "static/graph/"
socketio=SocketIO(app)
@app.route('/')
def index():

    actual_chart = go.Scatter (x=idr_df["Date"], y=idr_df["Units per EUR"],hovertemplate='%{y:.2f}', name='Data')

    with TemporaryDirectory() as tmp_dir:
        filename = tmp_dir + "tmp.html"
        py.plot([actual_chart],filename = filename, auto_open=False)
        with open(filename, "r") as f:
            graph_html = f.read()

        IS_FORECAST = False
        return render_template("index.html", price_day1=price_day1, change_1=change_1, change_7=change_7, change_14=change_14, change_31=change_31, graph_html=graph_html, IS_FORECAST=IS_FORECAST)


@app.route('/submit',methods=['POST'])
def submit_data():
    try:
        s2=int(request.form['parameter'])
        s1=request.form['options']
    except:
        flash("Please provide valid inputs")
        return redirect("/")

    new_idr = pd.read_csv('idr.csv', usecols=["Date", "Units per EUR"])
    new_idr['Date']= pd.to_datetime(new_idr.Date)
    data=new_idr[-s2:]

    x=np.array(range(len(new_idr)))
    x=x.reshape(-1,1)

    new_idr['day']=x+1
    x=new_idr.drop(['Date','Units per EUR'],axis = 1)
    y=new_idr['Units per EUR']
    features = x.columns

    loaded_model = joblib.load('model_SVR.sav')

    oot = pd.DataFrame(pd.date_range(end_date, periods=s2, freq=s1))
    oot.rename(columns={0:'Date'}, inplace=True)
    j = 1
    for i in range(len(oot)) :
        oot.loc[i, 'day'] = len(x)+j
        j=j+i
    
    new_data = data.append(oot)

    for i in range(len(oot)) :
        new_data.loc[i, 'Units per EUR'] = loaded_model.predict(new_data[new_data.index==i][features])
    
    oot_new=new_data
     
    forecast_data_orig = oot_new
    final_df=pd.DataFrame(forecast_data_orig)
    final_df['Date']=  pd.DataFrame(pd.date_range(end_date, periods=s2, freq=s1))
    
    X= pd.concat([data['Date'], final_df['Date']], ignore_index=True)
    Y = pd.concat([data['Units per EUR'], final_df['Units per EUR']], ignore_index=True)

    actual_chart = go.Scatter (x=X[-s2:],y=Y[-s2*2:],hovertemplate='%{y:.2f}',mode='lines+markers', name='Actual')
    predict_chart = go.Scatter(x=X[-s2:],y=Y[-s2:],hovertemplate='%{y:.2f}', mode='lines+markers', name='Predicted')
    

    with TemporaryDirectory() as tmp_dir:
        filename= tmp_dir + "tmp.html"
        py.plot([actual_chart,predict_chart],filename = filename, auto_open=False)
        with open(filename, "r") as f :
            graph_html= f.read()
    if s1=="D":
        value="Days"

    final_df_1=final_df[[ 'Date', 'Units per EUR']].tail(s2)
    final_df_1=final_df_1.rename(columns={'Date': 'Tanggal', 'Units per EUR': 'Prediksi Nilai Kurs (Rp)'})
    final_df_1.reset_index(drop=True, inplace=True)
    IS_FORECAST = True

    table = final_df_1.to_html(classes='table table-striped', border=0)
    table = table.replace('tr style="text-align: right;"', 'tr style="text-align: center;"')
    table = table.replace('<th></th>', '')
    table = table.replace('<th>', '<th colspan="2">', 1)
    print(table)
    return render_template("index.html",price_day1=price_day1,change_1=change_1,change_7=change_7,change_14=change_14,change_31=change_31, graph_html=graph_html, parameter=s2,table=table, IS_FORECAST = IS_FORECAST)


   

    
if __name__ =="__main__":


    socketio.run(app, port=8000, debug=True)
