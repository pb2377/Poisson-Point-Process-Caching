import glob
import os.path as osp

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from .cluster_expofit import ClusterAndExpoFit
from .utils import get_feed_id


class IntervalEstimator(ClusterAndExpoFit):
    def __init__(self, threshold=0):
        super().__init__(threshold)

    def __call__(self, feed_directory, key_to_use, classifier_threshold):
        return self.predict_dir_of_feeds(feed_directory, key_to_use, classifier_threshold)

    def predict_dir_of_feeds(self, feed_directory, key_to_use, classifier_threshold):
        output_df = {
            'FeedIdentifier': [],
            'CachingInterval': []
        }

        file_list = glob.glob(osp.join(feed_directory, '*'))
        for file_path in file_list:
            feed_id = get_feed_id(file_path)
            caching_interval = self.predict_one_feed(pd.read_csv(file_path), key_to_use)
            output_df['CachingInterval'].append(caching_interval)
            output_df['FeedIdentifier'].append(feed_id)

        output_df = pd.DataFrame(output_df)
        # classify if considered Cached
        output_df['Caching?'] = output_df['CachingInterval'] >= classifier_threshold
        return output_df

    def fit(self, training_data_dir, tuning_metric, classifier_threshold):
        # generate groundtruth labels from the time_published
        groundtruth_df = self(training_data_dir, 'time_published', classifier_threshold)

        perf_df = {
            'prec': [],
            'rec': [],
            'f1_score': [],
            't': []
        }

        for threshold in range(0, 60, 1):
            self.threshold = threshold
            preds_df = self.__call__(training_data_dir, 'time_created', classifier_threshold)
            cls_perf = precision_recall_fscore_support(groundtruth_df['Caching?'], preds_df['Caching?'],
                                                       average='weighted')
            perf_df = self._update_perf_dict(perf_df, cls_perf, threshold)

        # select best threshold performance
        perf_df = pd.DataFrame(perf_df)
        optimal_threshold = perf_df['t'][perf_df[tuning_metric].argmax()]
        self.threshold = optimal_threshold

    @staticmethod
    def _update_perf_dict(perf_df, cls_perf, threshold):
        prec, rec, f1_score, support = cls_perf

        # update perfomance dataframe
        perf_df['prec'].append(prec)
        perf_df['rec'].append(rec)
        perf_df['f1_score'].append(f1_score)
        perf_df['t'].append(threshold)
        return perf_df
