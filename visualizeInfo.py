#module for visualizing Information about the Dataframe to give a quick overview about most important things

import matplotlib.pyplot as plt
import numpy as np
def visualize(n_rows,n_cols,object_cols, numeric_cols, namemissing, missing):

    plt.style.use('seaborn')
    plt.figure(figsize=(10,6))

    plt.bar(namemissing,missing,width=0.8, align='center')
    plt.title('Information about Dataframe')
    for i in range (len(namemissing)):      #ha="center",va="bottom"
        plt.text(i,missing[i],missing[i],ha="center",va="bottom")
    plt.xticks(rotation='vertical',fontsize=5)
    plt.xlabel('columns')
    plt.ylabel('missing values')

    print(len(numeric_cols))
    y=np.array([len(numeric_cols),len(object_cols)])
    str1 = 'numeric: ',y[0]
    str2 = 'object: ',y[1]
    my_labels = str1, str2
    my_colors = ("orange","green")
    plt.figure(figsize=(3,3))
    plt.title('column_type')
    plt.pie(y,labels=my_labels,colors = my_colors)
    plt.show()

def plot_maes(maes, n_estimators):
    plt.style.use('seaborn')
    plt.figure(figsize=(10, 6))
    plt.xticks(rotation='vertical')
    plt.plot(n_estimators,maes)
    plt.title('mean absolut error for n_estimators')
    plt.xlabel('n_estimators')
    plt.ylabel('mean absolut error')
    plt.show()



