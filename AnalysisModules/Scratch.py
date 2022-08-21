def PerformanceMetrics(Type, NumberOfSplits):
    # dict constructor for a variety of performance metrics
    # Methods to calculate secondary & tertiary measures from primary measures

    # construct
    ModelPerformance = dict()

    # Metrics tiered by Most Common, Common, Uncommon
    if Type == "Regression":
        _common_metrics_list = ['r2', 'mean_absolute_error', 'mean_squared_error']
        # Common
        # r2 = None # R^2 /  Coefficient of Determination
        # mean_absolute_error = None # Mean Absolute Error

        _uncommon_metrics_list = ['median_absolute_error', 'max_error', 'explained_variance_score']
        # Uncommon
        # median_absolute_error = None # Median Absolute Error
        # max_error = None # Max Error
        # mean_squared_error = None  # Mean Squared Error
        # explained_variance_score = None  # Explained Variance Regression Score

        # Now I append. I realize this was kinda pointless, but it makes sense
        # purely from a documentation perspective. It seems too much work to

        _metrics_list = _common_metrics_list + _uncommon_metrics_list
        if NumberOfSplits == 2:
            _tuples_list = ['training', 'testing']
        elif NumberOfSplits == 3:
            _tuples_list = ['training', 'testing', 'validation']
        else:
            print("More than three splits is not yet implemented.")
            print("Using default train/test")
            _tuples_list = ['training', 'testing']
        _combined = list(itertools.product(_metrics_list, _tuples_list))

        for key in _combined:
            ModelPerformance.fromkeys(key, None)
            return ModelPerformance

    elif Type == "Classification":

        _most_common_metrics_list = ['accuracy', 'precision', 'recall', 'f1']
        # self.accuracy = None  # (tp+tn)/(tp+fn+fp+tn)
        # self.precision = None  # PPV or Precision, = tn/(tp+fp)
        # self.recall = None  # Sensitivity, TPR, Recall, = tp/(tp+fn)
        # self.f1 = None # F-Score, F-Measure

        _common_metrics_list = ['balanced_accuracy', 'AUC', 'AUC_PR']
        # self.balanced_accuracy = None  # Balanced Accuracy
        # self.AUC = None  # Area Under the Curve of Receiver Operating Characteristic
        # ROC = TPR vs. FPR
        # self.AUC_PR = None  # Area Under the Curve of Precision-Recall Curve

        _uncommon_metrics_list = ['specificity', 'fpr', 'fnr', 'tp', 'fn', 'fp', 'tn', 'rpp', 'rnp',
                                  'ecost', 'markedness', 'informedness']
        # self.specificity = None  # TNR, Specificity, = tn/(tn+fp)
        # self.fpr = None  # FPR, Fallout, = fp/(tn+fp)
        # self.fnr = None  # FNR, Miss, = fn/(tp+fn)
        # self.tp = None  # True Positive
        # self.fn = None  # False Negative
        # self.fp = None  # False Positive
        # self.tn = None  # True Negative
        # self.rpp = None  # Rate of Positive Predictions
        # = (tp+fp)/(tp+fn+fp+tn)
        # self.rnp = None  # Rate of Negative Predictions
        # = (tn+fn) / (tp+fn+fp+tn)
        # self.ecost = None # Expected Cost
        # ecost = (tp*Cost(P|P)+fn*Cost(N|P)+fp* Cost(P|N)+tn*Cost(N|N))/(tp+fn+fp+tn)
        # self.markedness = None # Markedness
        # self.informedness = None # Informedness

        _metrics_list = _most_common_metrics_list + _common_metrics_list + _uncommon_metrics_list
        if NumberOfSplits == 2:
            _tuples_list = ['training', 'testing']
        elif NumberOfSplits == 3:
            _tuples_list = ['training', 'testing', 'validation']
        else:
            print("More than three splits is not yet implemented.")
            print("Using default train/test")
            _tuples_list = ['training', 'testing']
        _combined = list(itertools.product(_metrics_list, _tuples_list))

        for key in _combined:
            ModelPerformance.fromkeys(key, None)
            return ModelPerformance