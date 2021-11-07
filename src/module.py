import os
import os.path as osp

from .interval_estimator import IntervalEstimator


class RSSCacheDetector:
    def __init__(self, interval_estimator=IntervalEstimator(), classifier_threshold=600, tuning_metric='f1_score'):
        self.interval_estimator = interval_estimator
        self.classifier_threshold = classifier_threshold
        self.tuning_metric = tuning_metric

    def __call__(self, target_directory, key_to_use):
        return self.predict(target_directory, key_to_use)

    def predict(self, target_directory, key_to_use):
        # Estimate systematic noise/latency of polling
        print('\nProcessing feeds from directory: {}'.format(target_directory))
        print('Estimating caching interval from {}...'.format(key_to_use))
        output_dataframe = self.interval_estimator(target_directory, key_to_use, self.classifier_threshold)

        save_path = self._make_save_path(target_directory, key_to_use)

        print('Saving outputs .csv to {}'.format(save_path))
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        output_dataframe.to_csv(save_path)
        return output_dataframe

    def fit(self, training_data_dir):
        self.interval_estimator.fit(training_data_dir, self.tuning_metric, self.classifier_threshold)

    @staticmethod
    def _make_save_path(target_directory, key_to_use):
        save_path = osp.basename(target_directory).lower() + '_intervals_from_' + key_to_use + '.csv'
        save_path = osp.join('outputs', save_path)
        return save_path
