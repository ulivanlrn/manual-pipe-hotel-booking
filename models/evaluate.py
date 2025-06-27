from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(y_true_train, y_pred_train, y_true_test, y_pred_test):
    """
    Function to compute evaluation metrics for train and test sets.
    :param y_true_train:
    :param y_pred_train:
    :param y_true_test:
    :param y_pred_test:
    :return: Dictionary with computed metrics.
    """
    scores = {}

    # balanced accuracy
    b_acc_train = round(balanced_accuracy_score(y_true_train, y_pred_train), 4)
    b_acc_test = round(balanced_accuracy_score(y_true_test, y_pred_test), 4)

    # precision
    pr_train = round(precision_score(y_true_train, y_pred_train), 4)
    pr_test = round(precision_score(y_true_test, y_pred_test), 4)

    # recall
    rec_train = round(recall_score(y_true_train, y_pred_train), 4)
    rec_test = round(recall_score(y_true_test, y_pred_test), 4)

    # f1-score
    f1_train = round(f1_score(y_true_train, y_pred_train), 4)
    f1_test = round(f1_score(y_true_test, y_pred_test), 4)

    scores['balanced accuracy train score'] = b_acc_train
    scores['precision train score'] = pr_train
    scores['recall train score'] = rec_train
    scores['f1 train score'] = f1_train

    scores['balanced accuracy test score'] = b_acc_test
    scores['precision test score'] = pr_test
    scores['recall test score'] = rec_test
    scores['f1 test score'] = f1_test

    return scores