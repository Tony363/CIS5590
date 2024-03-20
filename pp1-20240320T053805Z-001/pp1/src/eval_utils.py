# DO NOT IMPORT ANYTHING IN THIS FILE. You shouldn't need any external libraries.

# accuracy
#
# What percent of classifications are correct?
# 
# true: ground truth, Python list of booleans.
# pred: model predictions, Python list of booleans.
# return: percent accuracy bounded between [0, 1]
#
def accuracy(y_true, y_pred):
    acc = None
    ## YOUR CODE STARTS HERE (~2-5 lines of code) ##
    correct_predictions = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    total_predictions = len(y_true)
    return correct_predictions / total_predictions
    ## YOUR CODE ENDS HERE ##

# binary_f1 
#
# A method to calculate F-1 scores for a binary classification task.
# 
# args -
# true: ground truth, Python list of booleans.
# pred: model predictions, Python list of booleans.
# selected_class: Boolean - the selected class the F-1 
#                 is being calculated for.
# 
# return: F-1 score between [0, 1]
#
def binary_f1(y_true, y_pred, selected_class=True):
    f1 = None
    ## YOUR CODE STARTS HERE (~10-15 lines of code) ##
    TP = sum(1 for true, pred in zip(y_true, y_pred) if true == pred == 1)
    FP = sum(1 for true, pred in zip(y_true, y_pred) if pred == 1 and true != pred)
    FN = sum(1 for true, pred in zip(y_true, y_pred) if pred == 0 and true != pred)
    
    precision = TP / (TP + FP) if TP + FP else 0
    recall = TP / (TP + FN) if TP + FN else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall else 0
    return f1
    ## YOUR CODE ENDS HERE ##

# binary_macro_f1
# 
# Averaged F-1 for all selected (true/false) clases.
#
# args -
# true: ground truth, Python list of booleans.
# pred: model predictions, Python list of booleans.
#
#
def binary_macro_f1(y_true, y_pred):
    averaged_macro_f1 = None
    ## YOUR CODE STARTS HERE (1 line of code) ##
        # Calculating F1 for the positive class
    TP = sum(1 for true, pred in zip(y_true, y_pred) if true == pred == 1)
    FP = sum(1 for true, pred in zip(y_true, y_pred) if pred == 1 and true != pred)
    FN = sum(1 for true, pred in zip(y_true, y_pred) if pred == 0 and true != pred)
    precision_pos = TP / (TP + FP) if TP + FP else 0
    recall_pos = TP / (TP + FN) if TP + FN else 0
    f1_pos = 2 * (precision_pos * recall_pos) / (precision_pos + recall_pos) if precision_pos + recall_pos else 0
    
    # Calculating F1 for the negative class
    TN = sum(1 for true, pred in zip(y_true, y_pred) if true == pred == 0)
    FP = sum(1 for true, pred in zip(y_true, y_pred) if pred == 0 and true != pred)
    FN = sum(1 for true, pred in zip(y_true, y_pred) if pred == 1 and true != pred)
    precision_neg = TN / (TN + FN) if TN + FN else 0
    recall_neg = TN / (TN + FP) if TN + FP else 0
    f1_neg = 2 * (precision_neg * recall_neg) / (precision_neg + recall_neg) if precision_neg + recall_neg else 0
    
    # Macro F1 is the average of F1 scores for both classes
    macro_f1 = (f1_pos + f1_neg) / 2
    return macro_f1
    ## YOUR CODE ENDS HERE ##
