import numpy as np
import pandas as pd
from sklearn import metrics
from matplotlib import pyplot as plt
from save_load import *

matrices1=load('pred1')
matrices2=load('pred2')
matrices2=matrices2[~np.isnan(matrices2)]
num=0
matrices2=np.append(matrices2,[num])
matrices=np.column_stack([matrices1,matrices2])
matrices=pd.DataFrame(matrices)
#save('matrices',matrices)

def bar_plot(label,data,metric):
    df=pd.DataFrame(data)
    df1=pd.DataFrame()
    df1['dataset']=[1,2]
    df=pd.concat((df,df1),axis=1)
    df.plot(x='dataset',kind='bar',stacked=False)
    plt.ylabel(metric)
    plt.savefig('./Results/' + metric + '.png', dpi=400)
    plt.show(block=False)

def bar():

    matrices=np.array(load('matrices'))
    mthod=['ANN', 'CNN']
    metrices_plot=['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F-Measure', 'MCC', 'NPV', 'FPR', 'FNR']

    # Bar plot
    for i in range(len(metrices_plot)):
        bar_plot(mthod, matrices[i], metrices_plot[i])


    print('Testing Metrices-Dataset' )
    tab=pd.DataFrame(matrices, index=metrices_plot, columns=mthod)
    print(tab)


    Y_pred = load('ann_y_pred')
    Y_test=load('Y_test')
    auc = metrics.roc_auc_score(Y_test, Y_pred)

    false_positive_rate, true_positive_rate, thresolds = metrics.roc_curve(Y_test, Y_pred)

    plt.figure(figsize=(10, 8), dpi=100)
    plt.axis('scaled')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title("AUC & ROC Curve")
    plt.plot(false_positive_rate, true_positive_rate, 'g')
    plt.fill_between(false_positive_rate, true_positive_rate, facecolor='lightgreen', alpha=0.7)
    plt.text(0.95, 0.05, 'AUC = %0.4f' % auc, ha='right', fontsize=12, weight='bold', color='blue')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig('./Results/roc.png', dpi=400)
    plt.show()

bar()






