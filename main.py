from src import RSSCacheDetector

if __name__ == '__main__':
    # Estimate using time published
    rss_cache_detector = RSSCacheDetector()

    # training data the account article with normal publish/created times
    normal_publish_times_directory = None
    rss_cache_detector.predict(normal_publish_times_directory, key_to_use='time_published')

    # # Fit clustering threshold
    # already have a fitted value with F1_score to be threshold = 25
    rss_cache_detector.interval_estimator.threshold = 25

    # # # can generate estimations from time created for normal_publish_times_directory too

    # classify and predict the
    bad_publish_times_directory = None
    rss_cache_detector.predict(bad_publish_times_directory, key_to_use='time_created')
