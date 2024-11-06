import scienceplots
import matplotlib.pyplot as plt
plt.style.use(['science'])
import numpy as np
import itertools
import pandas as pd
import scienceplots
import matplotlib.pyplot as plt
plt.style.use(['science'])
import numpy as np
import pandas as pd

def estimate_confusion_matrix(sample_number, true_up_pct, pred_up_pct, accuracy):
    """
    Estimate confusion matrix based on available information.
    """
    total = sample_number
    true_up = int(total * true_up_pct)
    true_down = total - true_up
    pred_up = int(total * pred_up_pct)
    pred_down = total - pred_up
    
    correct = int(total * accuracy)
    
    # True Positives: correctly predicted ups
    tp = int(true_up * accuracy)
    
    # True Negatives: correctly predicted downs
    tn = correct - tp
    
    # False Positives: incorrectly predicted ups
    fp = pred_up - tp
    
    # False Negatives: incorrectly predicted downs
    fn = true_up - tp
    
    return np.array([[tn, fp], [fn, tp]])

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
        fmt = '.2f'
    else:
        print('Confusion matrix, without normalization')
        fmt = '.0f'

    print(cm)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Read CSV file
df = pd.read_csv('RIPT/WORK_DIR/log/20d20p_vbTrue_maTrue_oos_up_prob.csv', index_col=0)

# Separate IS and OOS data
is_data = df[df.index < '2018']
oos_data = df[df.index >= '2018']

# Calculate confusion matrices
cm_is = sum(estimate_confusion_matrix(row['Sample Number'], row['True Up Pct'], row['Pred Up Pct'], row['Accy']) 
            for _, row in is_data.iterrows())
cm_oos = sum(estimate_confusion_matrix(row['Sample Number'], row['True Up Pct'], row['Pred Up Pct'], row['Accy']) 
             for _, row in oos_data.iterrows())

# Define class labels
class_names = ['Down', 'Up']

# Plot confusion matrices
plot_confusion_matrix(cm_is, classes=class_names, normalize=False, 
                      title='IS Data Estimated Confusion Matrix')
plot_confusion_matrix(cm_is, classes=class_names, normalize=True, 
                      title='IS Data Estimated Normalized Confusion Matrix')

plot_confusion_matrix(cm_oos, classes=class_names, normalize=False, 
                      title='OOS Data Estimated Confusion Matrix')
plot_confusion_matrix(cm_oos, classes=class_names, normalize=True, 
                      title='OOS Data Estimated Normalized Confusion Matrix')

# Calculate and print metrics
def calculate_metrics(cm):
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return accuracy, precision, recall, f1

accuracy_is, precision_is, recall_is, f1_is = calculate_metrics(cm_is)
accuracy_oos, precision_oos, recall_oos, f1_oos = calculate_metrics(cm_oos)

print("In-Sample (IS) Metrics:")
print(f'Accuracy: {accuracy_is:.4f}')
print(f'Precision: {precision_is:.4f}')
print(f'Recall: {recall_is:.4f}')
print(f'F1 Score: {f1_is:.4f}')

print("\nOut-of-Sample (OOS) Metrics:")
print(f'Accuracy: {accuracy_oos:.4f}')
print(f'Precision: {precision_oos:.4f}')
print(f'Recall: {recall_oos:.4f}')
print(f'F1 Score: {f1_oos:.4f}')