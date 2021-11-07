import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def make_plots(dataframe, title_=''):
    dataframe['labels_str'] = dataframe['labels'].astype(str)
    dataframe['preds_str'] = dataframe['label_preds'].astype(str)

    values = dataframe['labels_str'].tolist() + dataframe['preds_str'].tolist()
    tags = ['gt'] * dataframe.shape[0] + ['pred'] * dataframe.shape[0]
    new_df = pd.DataFrame({
        'tag': tags,
        'vals': values
    }
    )

    sns.histplot(data=new_df, x='vals', hue='tag', multiple='dodge')
    plt.xlabel('Caching Label')
    plt.title(title_)
    plt.show()

    cf_matrix = confusion_matrix(dataframe['labels'], dataframe['label_preds'])
    cf_matrix_vals = cf_matrix.reshape(-1)  # .tolist()
    labels = ['True Neg\n{}', 'False Pos\n{}', 'False Neg\n{}', 'True Pos\n{}']
    labels = [i.format(cf_matrix_vals[idx]) for idx, i in enumerate(labels)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues',
                xticklabels=['False', 'True'], yticklabels=['False', 'True'])
    plt.xlabel('Predicted Caching Label')
    plt.ylabel('True Caching Label')
    plt.title(title_)
    plt.show()


def get_feed_id(file_path):
    feed_id = str(osp.basename(file_path).split('.')[0])
    return feed_id.split('_')[-1]
