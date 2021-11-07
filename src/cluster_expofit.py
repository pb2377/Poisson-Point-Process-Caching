import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import expon


class ClusterAndExpoFit:
    def __init__(self, threshold=0):
        self.threshold = threshold

    def predict_one_feed(self, feed_df, key_to_use, feed_id=None):
        # check all the article ids are unique
        assert len(set(feed_df['account_article_source_id'])) == feed_df.shape[0]
        if feed_df.shape[0] == 0:
            return 0

        feed_df = self._convert_unix_to_datetime(feed_df)
        caching_interval = self._fit_pdf(feed_df, key_to_use=key_to_use, feed_id=feed_id)
        return caching_interval

    def _fit_pdf(self, feed_df, key_to_use, feed_id):
        feed_df = self._sort_feed_by_time_created(feed_df, key_to_use)
        # only news to use the time created
        caching_intervals = self._compute_all_intervals(feed_df[key_to_use].tolist())
        if len(caching_intervals) > 0:
            params = expon.fit(caching_intervals)
            self._plot_pdf(caching_intervals, expon, params, feed_id, key_to_use)
            return params[0]
        else:
            # the feed is empty, so no value can be predicted
            return 0

    def _compute_all_intervals(self, time_stamps):
        time_stamps = self._compute_relative_times(time_stamps)
        # cluster time stamps based on threshold
        time_stamps = self._cluster_time_stamps(time_stamps)

        caching_intervals = []
        for idx, time_j in enumerate(time_stamps):
            caching_intervals.append(time_j - time_stamps[max(0, idx - 1)])

        caching_intervals = np.array(caching_intervals)
        # remove intervals that == 0, as they were published at the same time as another article
        return caching_intervals[caching_intervals > 0]

    def _cluster_time_stamps(self, time_stamps):
        clusters = {0: []}
        cluster_starts = {0: time_stamps[0]}

        target_cluster = 0
        for idx, time_ in enumerate(time_stamps):
            delta_t = time_ - cluster_starts[target_cluster]
            # start new cluster if above polling interval
            if delta_t > self.threshold:
                target_cluster += 1
                clusters[target_cluster] = [idx]
                cluster_starts[target_cluster] = time_
            else:
                # add to existing clusters otherwise
                clusters[target_cluster].append(idx)
                cluster_starts[target_cluster] = self._cluster_mean(clusters[target_cluster], time_stamps)
        return list(cluster_starts.values())

    @staticmethod
    def _cluster_mean(cluster_items, time_created):
        cluster_base = time_created[cluster_items[0]]
        cluster_times = [time_created[i] - cluster_base for i in cluster_items]
        return cluster_base + np.mean(cluster_times)

    @staticmethod
    def _convert_unix_to_datetime(df):
        for i in ['time_created', 'time_published']:
            df[i] = pd.to_datetime(df[i], unit='s')
        return df

    @staticmethod
    def _sort_feed_by_time_created(df, key):
        return df.sort_values(by=key, ascending=True)

    @staticmethod
    def _compute_relative_times(time_created):
        relative_time_created = np.array([(i - time_created[0]).total_seconds() for i in time_created])
        return relative_time_created

    @staticmethod
    def _plot_pdf(caching_intervals, pdf_fn, params, feed_id, key_to_use):
        if feed_id is not None:
            xx = np.arange(np.max(caching_intervals))
            yy = pdf_fn.pdf(xx, *params)
            plt.hist(caching_intervals, density=True)
            plt.plot(xx, yy)
            plt.title('Feed ID {} ({}): Estimated caching interval {}'.format(feed_id, key_to_use, params[0]))
            plt.show()
