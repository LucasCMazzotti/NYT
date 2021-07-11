#data stuff
import time
import pandas as pd
import numpy as np
import datetime as dt

#regression stuff
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

#graph stuff
import seaborn as sns
from plotly.subplots import make_subplots
from plotly import graph_objects as go
import plotly
import matplotlib.pyplot as plt

def plot_variables_func(df,plot_list, one_plot=False, title=None, monthly=False):
    if monthly == False:
        x = pd.date_range(df.index.min(), df.index.max())
        y_df = df.copy()
        y_df = y_df.reindex(x)
        plot_me = plot_list
        data = []
        palette =[
        '#1f77b4',  # muted blue
        '#ff7f0e',  # safety orange
        '#7f7f7f',  # middle gray
        '#2ca02c',  # cooked asparagus green
        '#d62728',  # brick red
        '#9467bd',  # muted purple
        '#8c564b',  # chestnut brown
        '#e377c2', # raspberry yogurt pink
        '#1f77b4',  # muted blue
        '#ff7f0e',  # safety orange
        '#7f7f7f',  # middle gray
        '#2ca02c',  # cooked asparagus green
        '#d62728',  # brick red
        '#9467bd',  # muted purple
        '#8c564b',  # chestnut brown
        '#e377c2', # raspberry yogurt pink
        '#1f77b4',  # muted blue
        '#ff7f0e',  # safety orange
        '#7f7f7f',  # middle gray
        '#2ca02c',  # cooked asparagus green
        '#d62728',  # brick red
        '#9467bd',  # muted purple
        '#8c564b',  # chestnut brown
        '#e377c2', # raspberry yogurt pink
        '#1f77b4',  # muted blue
        '#ff7f0e',  # safety orange
        '#7f7f7f',  # middle gray
        '#2ca02c',  # cooked asparagus green
        '#d62728',  # brick red
        '#9467bd',  # muted purple
        '#8c564b',  # chestnut brown
        '#e377c2', # raspberry yogurt pink
        ]

        for i in range(len(plot_me)):
            # Create a trace
            trace = go.Scatter(
                x = df.index,
                y = df.loc[:,plot_me[i]],
                name = plot_me[i],
                line = dict(color = palette[i])
            )

            if one_plot == False:
                data = [trace]
                layout=go.Layout(title=plot_me[i])

                plotly.offline.iplot(go.Figure(data=data, layout=layout), filename='basic-line')
            else:
                data.append(trace)

        if one_plot == True:
            layout=go.Layout(title=title,
                            legend=dict(orientation='h',xanchor = "center",x = 0.5))
            plotly.offline.iplot(go.Figure(data=data, layout = layout), filename='scatter-mode')
    else:
        aggre = {plot_list[i]:'sum' for i in range(len(plot_list))}
        if isinstance(df.index, pd.DatetimeIndex) == False:
            df_m = df.set_index('dte').groupby(pd.Grouper(freq='M')).agg(aggre)
        else:
            df_m = df.groupby(pd.Grouper(freq='M')).agg(aggre)

        plot_me = plot_list
        data = []
        for i in range(len(plot_me)):
            # Create a trace
            trace = go.Scatter(
                x = df_m.index,
                y = df_m.loc[:,plot_me[i]],
                name = plot_me[i]
            )

            if one_plot == False:
                data = [trace]
                layout=go.Layout(title=plot_me[i])
                plotly.offline.iplot(go.Figure(data=data, layout=layout), filename='basic-line')
            else:
                data.append(trace)

        if one_plot == True:
            plotly.offline.iplot(go.Figure(data=data), filename='scatter-mode')

            
def plot_variables(df, plot_list, one_plot=False, title=None, monthly=False, scaled=False):
    if scaled == True:
        df_scale = df[plot_list]
        for col in df_scale.columns:
            x = df_scale[[col]].values.astype(float)
            min_max_scaler = MinMaxScaler(feature_range=(0,1))
            x_scaled = min_max_scaler.fit_transform(x)
            df_scale[col] = x_scaled
        plot_variables_func(df_scale, plot_list, one_plot=one_plot, title=title, monthly=monthly)
    else:
        plot_variables_func(df, plot_list, one_plot=one_plot, title=title, monthly=monthly)
        

def get_corr_matrix(df, var_list, start, end, fontsize=12):

#     label_list = [x.partition('_adstock')[0] for x in var_list]
#     label_list = [x.partition('_impressions')[0] for x in label_list]
#     label_list = [x.replace("fixed"," ") for x in label_list]
#     label_list = [x.replace("_"," ").title() for x in label_list]
    label_list = var_list


    corr = df.loc[start:end][var_list].corr()

    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots(figsize=(15, 12))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(corr, 
#                 xticklabels=label_list, 
#                 yticklabels=label_list, 
                mask=mask, 
                cmap=cmap, 
                vmin=-1,
                vmax=1, 
                center=0,
                square=True, 
                linewidths=.5, 
                annot=True, 
                annot_kws={"size":fontsize}, 
                fmt='.1g'
               )
    plt.tight_layout()
    

def oos_testing(df,train_lm,vars_list,oos_start,oos_end,is_start,is_end,dep='starts'):
    oos_data = df[vars_list].loc[oos_start:oos_end]
    pred = train_lm.predict(oos_data)
    pred = pred.to_frame(name='predicted')
    actuals = df[dep].loc[oos_start:oos_end]
    actuals = actuals.to_frame(name='actuals')
    oos_df = pd.merge(pred,actuals,left_index=True,right_index=True)
    oos_df['resid'] = oos_df['predicted'] - oos_df['actuals']
    oos_df['abs_pct_error'] = abs(oos_df['predicted']/oos_df['actuals'] - 1)
    oos_mape = oos_df['abs_pct_error'].replace(np.inf, np.NAN).mean()

    is_data = df[vars_list].loc[is_start:is_end]
    pred = train_lm.predict(is_data)
    pred = pred.to_frame(name='predicted')
    actuals = df[dep].loc[is_start:is_end]
    actuals = actuals.to_frame(name='actuals')
    is_df = pd.merge(pred,actuals,left_index=True,right_index=True)
    is_df['resid'] = is_df['predicted'] - is_df['actuals']
    is_df['abs_pct_error'] = abs(is_df['predicted']/is_df['actuals'] - 1)
    is_mape = is_df['abs_pct_error'].replace(np.inf, np.NAN).mean()

    all_df = pd.concat([is_df, oos_df])
    all_mape = all_df.abs_pct_error.mean()

    print("Full Mape", round(all_mape,3))
    print("In-Sample Mape", round(is_mape,3))
    print("Out-of-Sample Mape", round(oos_mape,3))

    return all_df