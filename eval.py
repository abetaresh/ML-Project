"""

Author: Vladimir Despotovic
University of Luxembourg
Department of Computer Science

"""

import pandas
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# Load ground truth and predicted test labels
# Name your document with predicted labels as: 'test.eval.txt'
y_true = pandas.read_csv('../data/train/train.csv',delimiter=' ')
y_pred = pandas.read_csv('../save/results.csv',delimiter=' ')
y_true.shape
y_pred.shape

# Remove the first column (file name)
y_true = y_true.drop('file_name', axis=1)
y_pred = y_pred.drop('file_name', axis=1)

# Calculate precision, recall, and f1 score
precision = precision_score(y_true, y_pred, average=None)
recall = recall_score(y_true, y_pred, average=None)
f1 = f1_score(y_true, y_pred, average=None)

# Create the list of all classes
classes = list(y_true.columns)

# Prepare for printing
dash = '-' * 45
# Loop over classes
for i in range(len(classes)+1):
    # Print the header
    if i == 0:
      print(dash)
      print('{:<15}{:<12}{:<9}{:<4}'.format('Class','precision','recall','f1 score'))
      print(dash)
    # Print precision, recall and f1 score for each of the labels
    else:
      print('{:<17}{:<11.2f}{:<10.2f}{:<10.2f}'.format(classes[i-1],precision[i-1],recall[i-1],f1[i-1]))

# Print average precision
precision_micro = precision_score(y_true, y_pred, average='micro')
print('{:<20}{:<4.2f}'.format('\nAverage precision:',precision_micro))
# Print average recall
recall_micro = recall_score(y_true, y_pred, average='micro')
print('{:<19}{:<4.2f}'.format('Average recall:',recall_micro))
# Print average f1 score
f1_micro = f1_score(y_true, y_pred, average='micro')
print('{:<19}{:<12.2f}'.format('Average f1 score:',f1_micro))
