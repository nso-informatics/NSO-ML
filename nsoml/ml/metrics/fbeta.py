from sklearn.metrics import confusion_matrix


def fbeta(y_test, y_pred, beta=10):
    '''
    Function to calculate the F-beta score.
    
    Parameters:
    y_test: array-like, true labels
    y_pred: array-like, predicted labels
    beta: float, beta value
    
    Returns:
    fbeta_score: float, F-beta score
    '''
    
    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate precision and recall
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])
    recall = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    
    # Calculate F-beta score
    fbeta_score = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    
    return fbeta_score
