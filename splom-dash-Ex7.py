# -*- coding: utf-8 -*-
"""
Created on Fri May  3 15:07:13 2019

@author: Maryam Assaedi
@author: Mst. Mahfuja Akter
@author: Mahpara Hyder Chowdhury
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from sklearn.feature_selection import SelectPercentile, f_classif

import pandas as pd
from textwrap import dedent as d
import plotly.graph_objs as go
from plotly import tools
from dash.dependencies import Input, Output

import json
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
import numpy as np

def calculatePCA(selectedData):
    pca  = PCA(n_components=2).fit_transform(X)
    principalDf = pd.DataFrame(pca)
    principalDf = pd.concat([principalDf, finaldf[['class']]], axis = 1)
    principalDf.columns=['PCA1','PCA2','class']
    principalDf.reset_index(drop=True, inplace=True)
    classes = np.unique(finaldf['class'].values).tolist()
    #classes
    class_code = {classes[k]: k for k in range(2)}
    #class_code
    color_vals = [class_code[cl] for cl in finaldf['class']]
    pl_colorscale = [[0.0, 'red'], [0.5, 'red'], [0.5, 'blue'], [1, 'blue']]

    selectedPoints = principalDf.index
    if selectedData:
        listOfPoints = selectedData['points']
        listOfIndices = []
        for i in range(len(listOfPoints)):
            listOfIndices.append(listOfPoints[i]['pointIndex'])
        
        if len(listOfIndices) > 0:
            selectedPoints = np.intersect1d(selectedPoints,listOfIndices)

    
    return {'data': [go.Scatter(
            x=principalDf['PCA1'],
            y=principalDf['PCA2'],
            text = principalDf['class'],
            selectedpoints =  selectedPoints,
        mode='markers',
            marker={
                 'size': 7,
                'opacity': 0.8,
              'color': color_vals,
              'colorscale' :pl_colorscale,
              'showscale' :False
            }
        )],
        'layout': go.Layout(
            xaxis={
                'title': principalDf['PCA1'].name,
                'type': 'linear'
            },
            yaxis={
                'title': principalDf['PCA2'].name,
                'type': 'linear'
            },
            margin={'l': 40, 'b': 30, 't': 10, 'r': 0},
            height=450,
            hovermode='closest',
            clickmode = 'event+select'
                    )}


def calculatetSNE(selectedData):
    tsne = TSNE(n_components=2,perplexity=20).fit_transform(X)
    newTsneDf = pd.DataFrame(data = tsne, columns = ['TSNE1','TSNE2'])
    tsnedf = pd.concat([newTsneDf, finaldf[['class']]], axis = 1)
    classes = np.unique(finaldf['class'].values).tolist()
    #classes
    class_code = {classes[k]: k for k in range(2)}
    #class_code
    color_vals = [class_code[cl] for cl in finaldf['class']]

    pl_colorscale = [[0.0, 'red'], [0.5, 'red'], [0.5, 'blue'], [1, 'blue']]
    selectedPoints = tsnedf.index
    if selectedData:
        listOfPoints = selectedData['points']
        listOfIndices = []
        for i in range(len(listOfPoints)):
            listOfIndices.append(listOfPoints[i]['pointIndex'])
        
        if len(listOfIndices) > 0:
            selectedPoints = np.intersect1d(selectedPoints,listOfIndices)
       
    

    return {'data': [go.Scatter(
            x=tsnedf['TSNE1'],
            y=tsnedf['TSNE2'],
            text = tsnedf['class'],
            selectedpoints =  selectedPoints,
	    mode='markers',
            marker={
                 'size': 7,
                'opacity': 0.8,
              'color': color_vals,
              'colorscale' :pl_colorscale,
              'showscale' : False
            }
        )],
        'layout': go.Layout(
            xaxis={
                'title': tsnedf['TSNE1'].name,
                'type': 'linear'
            },
            yaxis={
                'title': tsnedf['TSNE2'].name,
                'type': 'linear'
            },
            margin={'l': 40, 'b': 30, 't': 10, 'r': 0},
            height=450,
            hovermode='closest',
            clickmode = 'event+select'
        )}

def calculateISOMAP(selectedData):
    iso  = Isomap(n_neighbors=5, n_components=2).fit_transform(X)
    newIsoDf = pd.DataFrame(data = iso, columns = ['ISOMAP1','ISOMAP2'])
    newIsoDf = pd.concat([newIsoDf, finaldf[['class']]], axis = 1)
    classes = np.unique(finaldf['class'].values).tolist()
    #classes
    class_code = {classes[k]: k for k in range(2)}
    #class_code
    color_vals = [class_code[cl] for cl in finaldf['class']]

    pl_colorscale = [[0.0, 'red'], [0.5, 'red'], [0.5, 'blue'], [1, 'blue']]

    selectedPoints = newIsoDf.index
    if selectedData:
        listOfPoints = selectedData['points']
        listOfIndices = []
        for i in range(len(listOfPoints)):
            listOfIndices.append(listOfPoints[i]['pointIndex'])
        
        if len(listOfIndices) > 0:
            selectedPoints = np.intersect1d(selectedPoints,listOfIndices)
    
    return {'data': [go.Scatter(
            x=newIsoDf['ISOMAP1'],
            y=newIsoDf['ISOMAP2'],
            text = newIsoDf['class'],
            selectedpoints =  selectedPoints,
	    mode='markers',
            marker={
                 'size': 7,
                'opacity': 0.8,
              'color': color_vals,
              'colorscale' :pl_colorscale,
              'showscale' :False
            }
        )],
        'layout': go.Layout(
            xaxis={
                'title': newIsoDf['ISOMAP1'].name,
                'type': 'linear'
            },
            yaxis={
                'title': newIsoDf['ISOMAP2'].name,
                'type': 'linear'
            },
            margin={'l': 40, 'b': 30, 't': 10, 'r': 0},
            height=450,
            hovermode='closest',
            clickmode = 'event+select'
        )}

def calculateMatrixAttributes(selectedData,selected):
    if selected == False:
        

        X = finaldf.iloc[:, 0:77].values
        y = finaldf['class'].values

        selector = SelectPercentile(f_classif)
        attr = selector.fit(X, y)
        scores = -np.log10(selector.pvalues_)
        scores /= scores.max()
        attr = np.argpartition(scores,-5)[-5:]
        
        tempdf = finaldf.iloc[:, attr]

        attributes=list(tempdf)
        tempdf.reset_index(drop=True, inplace=True)
        tempdf['class']=finaldf['class']
        
        data_c = tempdf[tempdf['class'] == 'c-SC-s']
        data_t = tempdf[tempdf['class'] == 't-SC-s']
    else:
        X = finaldf.iloc[:, 0:77].values
        y = finaldf['selected'].values

        selector = SelectPercentile(f_classif)
        attr = selector.fit(X, y)
        scores = -np.log10(selector.pvalues_)
        scores /= scores.max()
        attr = np.argpartition(scores,-5)[-5:]
        
        tempdf = finaldf.iloc[:, attr]

        attributes=list(tempdf)
        tempdf.reset_index(drop=True, inplace=True)
        tempdf['selected']=finaldf['selected']
        
        selected = tempdf[tempdf['selected'] == 1]
       
        not_selected = tempdf[tempdf['selected'] == 0]
        return(selected,not_selected,attributes)

    return (data_c,data_t,attributes)

df = pd.read_excel('Data_Cortex_Nuclear.xls')

df2 = df.fillna(0)

df3 = df2[df2.loc[:, 'class'].isin(df[df['class'] == 'c-SC-s']['class'])]
df3 = df3.append(df2[df2.loc[:, 'class'].isin(df[df['class'] == 't-SC-s']['class'])])
finaldf = df3.drop(['MouseID','Genotype','Treatment','Behavior'], axis=1)
finaldf.reset_index(drop=True, inplace=True)

X = finaldf.iloc[:, 0:77].values
y = finaldf['class'].values
protein_list = list(finaldf)
protein_list.remove('class')

figPCA=tools.make_subplots(rows=1,   cols=1)
figtSNE=tools.make_subplots(rows=1,   cols=1)
figISOMAP=tools.make_subplots(rows=1,   cols=1)

figScatter = tools.make_subplots(rows= 5, cols = 5)
figProtein = tools.make_subplots(rows=1,   cols=1)

show_legend = False

app = dash.Dash(__name__)
colors = {
    'background': '#115511',
    'text': '#7FDBFF'
}
app.layout = html.Div(children=[
    html.H1(
            children='Exercise 7',
            style={
            'textAlign': 'center',
            'color': colors['text']
            }
            ),

    html.Div(children='Interactive Visualization with Dash',
         style={
                 'textAlign': 'center',
        'color': 'green'
        }
    ), 
    html.Div([
        dcc.Dropdown(
            id='my-dropdown',
            options=[
                {'label': 'PCA', 'value': 'PCA'},
                {'label': 'tSNE', 'value': 'tSNE'},
                {'label': 'ISOMAP', 'value': 'ISOMAP'}
            ],
            value='PCA'
        ),
        dcc.Graph(
            id='outputGraph'
        )],style={'width': '48%','margin': '4px', 'float' : 'left' ,'display': 'inline-block'}),
    html.Div([
        dcc.Dropdown(
                id='my_protein_list1',
                options=[{'label': i, 'value': i} for i in protein_list],
                value=protein_list[0]
            ),
        dcc.Dropdown(
                id='my_protein_list2',
                options=[{'label': i, 'value': i} for i in protein_list],
                value=protein_list[1]
            ),
        dcc.Graph(
            id='outputGraph2'
            
        )],style={'width': '48%','margin': '4px', 'display': 'inline-block'}),
    html.Div([
        dcc.Graph(
            id='scatterplot',
            figure = figScatter            
        )],style={'width': '100%','margin': '4px', 'display': 'inline-block'}) 
])

@app.callback(
    Output('outputGraph','figure'),
    [Input('outputGraph', 'selectedData'),
    Input('my-dropdown', 'value')])
def update_output(selectedData,value):   
    if value == 'PCA':
        return calculatePCA(selectedData)
    elif value == 'tSNE':
        return calculatetSNE(selectedData)
    elif value == 'ISOMAP':
        return calculateISOMAP(selectedData)
    
@app.callback(
    Output('outputGraph2','figure'),
    [Input('outputGraph', 'selectedData'),
    Input('my_protein_list1', 'value'),
    Input('my_protein_list2', 'value')]
    )
def update_protein_graph(selectedData,value1,value2):
    proteinData = pd.concat([finaldf[value1], finaldf[value2], finaldf[['class']]], axis = 1)
    classes = np.unique(finaldf['class'].values).tolist()
    #classes
    class_code = {classes[k]: k for k in range(2)}
    #class_code
    color_vals = [class_code[cl] for cl in finaldf['class']]
    pl_colorscale = [[0.0, 'red'], [0.5, 'red'], [0.5, 'blue'], [1, 'blue']]
    selectedPoints = proteinData.index
    if selectedData:
        listOfPoints = selectedData['points']
        listOfIndices = []
        for i in range(len(listOfPoints)):
            listOfIndices.append(listOfPoints[i]['pointIndex'])
        
        if len(listOfIndices) > 0:
            selectedPoints = np.intersect1d(selectedPoints,listOfIndices)    

    return {
        'data': [go.Scatter(
            x=proteinData[value1],
            y=proteinData[value2],
	       text = proteinData['class'],
           selectedpoints =  selectedPoints,
           mode='markers',
            marker={
                'size': 7,
                'opacity': 0.8,
		      'color': color_vals,
              'colorscale' :pl_colorscale,
              'showscale' :False
            }
        )],
        'layout': go.Layout(
            xaxis={
                'title': proteinData[value1].name,
                'type': 'linear'
            },
            yaxis={
                'title': proteinData[value2].name,
                'type': 'linear'
            },
            margin={'l': 40, 'b': 30, 't': 10, 'r': 0},
            height=450,
            hovermode='closest'
        )
    }

@app.callback(
    Output('scatterplot','figure'),
    [Input('outputGraph', 'selectedData')]
    )
def scatterMatrix(selectedData):
    selectedPoints = finaldf.index
    figScatter = tools.make_subplots(rows= 5, cols = 5)
    show_legend=False
    if selectedData:
        
        listOfPoints = selectedData['points']
        listOfIndices = []
        for i in range(len(listOfPoints)):
            listOfIndices.append(listOfPoints[i]['pointIndex'])
        
        if len(listOfIndices) > 0:
            selectedPoints = np.intersect1d(selectedPoints,listOfIndices)
       
        finaldf['selected'] = 0
        finaldf.loc[selectedPoints,'selected']=1
       
        selected,not_selected,attributes= calculateMatrixAttributes(selectedPoints,True)
       
        for i in range(len(attributes)):
            for j in range(len(attributes)):
                #Place histograms on the diagonal
                if i==j:
                    #Only add the legend once (for the last plot)
                    if i ==  len(attributes)-1 and j == len(attributes)-1:
                        show_legend = True
                    figScatter.append_trace(go.Histogram(
                        x = selected[attributes[i]],
                        opacity = 0.8,
                        text='selected',
                        name = 'selected',
                        marker = dict(
                            color = 'purple'
                        ),
                        showlegend = show_legend,
                    ), i+1, j+1)
                    figScatter.append_trace(go.Histogram(
                        x = not_selected[attributes[i]],
                        opacity = 0.8,
                        text='not_selected',
                        name = 'not_selected',
                        marker = dict(
                            color = 'yellow'
                        ),
                        showlegend = show_legend,

                    ), i+1, j+1)
                else:
                    #Add the scatterplots
                    figScatter.append_trace(go.Scatter(
                        x = selected[attributes[i]],
                        y = selected[attributes[j]],
                        mode = 'markers',
                        text=['selected']*selected.shape[0],
                        opacity = 0.8,
                        marker = dict(
                            color = 'purple',
                        ),
                        name = 'selected',
                        showlegend=False,
                    ), i+1, j+1)
                    figScatter.append_trace(go.Scatter(
                        x = not_selected[attributes[i]],
                        y = not_selected[attributes[j]],
                        mode = 'markers',
                        text=['not_selected']*not_selected.shape[0],
                        opacity = 0.8,
                        marker = dict(
                            color = 'yellow'
                        ),
                        name = 'not_selected',
                        showlegend=False
                    ), i+1, j+1)
    else:
        data_c,data_t,attributes= calculateMatrixAttributes(selectedPoints,False)

        #Add plot to the grid
        for i in range(len(attributes)):
            for j in range(len(attributes)):
                #Place histograms on the diagonal
                if i==j:
                    #Only add the legend once (for the last plot)
                    if i ==  len(attributes)-1 and j == len(attributes)-1:
                        show_legend = True
                    figScatter.append_trace(go.Histogram(
                        x = data_c[attributes[i]],
                        opacity = 0.8,
                        text='c-SC-s',
                        name = 'c-SC-s',
                        marker = dict(
                            color = 'red'
                        ),
                        showlegend = show_legend,
                    ), i+1, j+1)
                    figScatter.append_trace(go.Histogram(
                        x = data_t[attributes[i]],
                        opacity = 0.8,
                        text='t-SC-s',
                        name = 't-SC-s',
                        marker = dict(
                            color = 'blue'
                        ),
                        showlegend = show_legend,

                    ), i+1, j+1)
                else:
                    #Add the scatterplots
                    figScatter.append_trace(go.Scatter(
                        x = data_c[attributes[i]],
                        y = data_c[attributes[j]],
                        mode = 'markers',
                        text=['c-SC-s']*data_c.shape[0],
                        opacity = 0.8,
                        marker = dict(
                            color = 'red',
                        ),
                        name = 'c-SC-s',
                        showlegend=False,
                    ), i+1, j+1)
                    figScatter.append_trace(go.Scatter(
                        x = data_t[attributes[i]],
                        y = data_t[attributes[j]],
                        mode = 'markers',
                        text=['t-SC-s']*data_t.shape[0],
                        opacity = 0.8,
                        marker = dict(
                            color = 'blue'
                        ),
                        name = 't-SC-s',
                        showlegend=False
                    ), i+1, j+1)

    #Define the layout 
    figScatter.layout.update(go.Layout(
        barmode = 'overlay',
        clickmode= 'event+select',
        height=700,
        #add x axis titles
         annotations=[
            dict(
                x=0.06,
                y=1.05,
                showarrow=False,
                text=attributes[0],
                xref='paper',
                yref='paper'
            ),
            dict(
                x=0.26,
                y=1.05,
                showarrow=False,
                text=attributes[1],
                xref='paper',
                yref='paper'
            ),
            dict(
                x=0.5,
                y=1.05,
                showarrow=False,
                text=attributes[2],
                xref='paper',
                yref='paper',
            ),
            dict(
                x=0.75,
                y=1.05,
                showarrow=False,
                text=attributes[3],
                xref='paper',
                yref='paper'
            ),
            dict(
                x=0.93,
                y=1.05,
                showarrow=False,
                text=attributes[4],
                xref='paper',
                yref='paper'
            )],

        #Add y axis titles
        yaxis1 = dict (
            title = attributes[0],
            titlefont=dict(
                size=12,
            )
        ),
        yaxis6 = dict (
            title = attributes[1],
            titlefont=dict(
                size=12,
            )
        ),
        yaxis11 = dict (
            title = attributes[2],
            titlefont=dict(
                size=12,
            )
        ),
        yaxis16 = dict (
            title = attributes[3],
            titlefont=dict(
                size=12,
            )
        ),
        yaxis21 = dict (
            title = attributes[4],
            titlefont=dict(
                size=12,
            )
        )
        ))
    return figScatter
if __name__ == '__main__':
    app.run_server(debug=True)
