import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import plotly.offline as py
import plotly.graph_objs as go
import plotly.io as pio
from sklearn.metrics import roc_curve

# === Static parameters ===
default_xaxis_dict = dict(
    tickangle=-90,
    tickfont=dict(
        size=9,
        color='black'
    ),
    ticklen=5,
)

margin_params = go.layout.Margin(
    b=220,
    l=50,
    r=50
)

default_height = 650
# ========================

def plot_feature_imp_for_classifier(feature_dataframe, clf_id, plot_title, superv_feat_imp):
    trace = go.Bar(
        y=feature_dataframe[clf_id].values,
        x=feature_dataframe['features'].values,
        width=0.5,
        # mode='markers',
        marker=dict(
            # sizemode='diameter',
            # sizeref=1,
            # size=25,
            color=feature_dataframe[clf_id].values,
            colorscale='Portland',
            showscale=True,
            reversescale=False
        ),
        text=feature_dataframe['features'].values
    )
    data = [trace]

    layout = go.Layout(
        autosize=True,
        height=default_height,
        title=plot_title,
        hovermode='closest',
        xaxis=default_xaxis_dict,
        yaxis=dict(
            title='Feature Importance',
            ticklen=5,
            gridwidth=2
        ),
        margin=margin_params,
        showlegend=False
    )
    fig = go.Figure(data=data, layout=layout)

    filename = clf_id + '_feature_imp'
    py.offline.plot(fig, filename=str(superv_feat_imp / (filename + '.html')), auto_open=False)



def plot_average_feature_importance_scatterplots(feature_dataframe, superv_feat_imp):
    if 'SVC feature importances' in feature_dataframe.columns.values:
        feature_dataframe = feature_dataframe.drop(['SVC feature importances'], axis=1)

    feature_dataframe['mean'] = feature_dataframe.mean(axis=1)  # axis = 1 computes the mean row-wise
    feature_dataframe.sort_values(by=['mean'], ascending=False).head(10)

    y = feature_dataframe['mean'].values
    x = feature_dataframe['features'].values
    data = [go.Bar(
        x=x,
        y=y,
        width=0.5,
        marker=dict(
            color=feature_dataframe['mean'].values,
            colorscale='Portland',
            showscale=True,
            reversescale=False
        ),
        opacity=0.6
    )]

    layout = go.Layout(
        autosize=True,
        height=default_height,
        title='Ensemble classifier - Mean Feature Importance',
        hovermode='closest',
        xaxis=default_xaxis_dict,
        yaxis=dict(
            title='Feature Importance',
            ticklen=5,
            gridwidth=2
        ),
        margin=margin_params,
        showlegend=False
    )
    fig = go.Figure(data=data, layout=layout)

    filename = 'Ensemble_Classifier.average_feature_imp.html'
    py.offline.plot(fig, filename=str(superv_feat_imp / filename), auto_open=False)


def plot_confusion_matrix(cm, classes,
                          normalise=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          pdf_filepath='./confusion_matrix.pdf',
                          verbose=False):
    '''
    Print and plot the confusion matrix 
    :param cm: 
    :param classes: 
    :param normalise: 
    :param title: 
    :param cmap: 
    :param pdf_filepath: 
    :return: 
    '''
    if normalise:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        if verbose:
            print("Normalised confusion matrix")
    elif verbose:
            print('Confusion matrix, without normalisation')
    # print(cm)

    font = {'weight': 'normal',
            'size': 16}
    matplotlib.rc('font', **font)
    matplotlib.rc('xtick', labelsize=12)
    matplotlib.rc('ytick', labelsize=12)

    f = plt.figure(figsize=(5, 5))
    _ = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #     plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalise else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    f.savefig(pdf_filepath) #, bbox_inches='tight')



if __name__ == '__main__':

    # test plotly offline
    trace = {'x':[1,2],'y':[1,2]}
    data = [trace]
    fig = go.Figure(data = data, layout = {})
    py.offline.plot(fig, filename='test.html', auto_open=False)
