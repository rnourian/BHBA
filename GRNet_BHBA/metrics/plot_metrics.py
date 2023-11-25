import pandas as pd
import matplotlib.pyplot as plt
from torchmetrics import ConfusionMatrix
from torchmetrics import MetricCollection
from mlxtend.plotting import plot_confusion_matrix
from torchmetrics import F1Score, Recall, Precision, Accuracy


def plot_confusion_mat(y_target, y_predicted, classes):
    conf = ConfusionMatrix(num_classes=len(classes), task='multiclass')
    conf_tensor = conf(y_predicted, y_target)
    fig1, ax = plot_confusion_matrix(conf_mat=conf_tensor.numpy(), class_names=classes, figsize=(10, 7), show_absolute=True, show_normed=True, fontcolor_threshold=0.8)
    plt.show(block=True)


def plot_table_of_metrics(y_target, y_predicted, classes):
    metric_coll = MetricCollection({
        'Accuracy': Accuracy(task="multiclass", num_classes=len(classes)),
        'Precision': Precision(task="multiclass", average='macro', num_classes=len(classes)),
        'Recall': Recall(task="multiclass", average='macro', num_classes=len(classes)),
        'F1 Score': F1Score(task="multiclass", average='macro', num_classes=len(classes))

    })
    metric_coll.update(y_predicted, y_target)
    metrics = metric_coll.compute()

    metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Metric'])
    metrics_df.index.name = 'Metric Type'
    metrics_df.reset_index(inplace=True)
    print(" ")
    print(metrics_df)


def plot_acc_graph(train_losses, total_train_acc, validation_losses, total_validation_acc, epoch):
    trn_loss = [fl.item() for fl in train_losses]
    trn_accuracy = [ac.item() for ac in total_train_acc]
    val_loss = [fl.item() for fl in validation_losses]
    val_accuracy = [ac.item() for ac in total_validation_acc]
    fig2, axes = plt.subplots(nrows=2, ncols=2)

    axes[0, 0].plot(range(epoch), trn_loss, linewidth=3.0)
    axes[0, 0].set_xlabel('Epochs ', fontsize=10)
    axes[0, 0].set_ylabel('Loss', fontsize=10)
    axes[0, 0].set_title('Training Loss Curve', fontsize=10)

    axes[0, 1].plot(range(epoch), trn_accuracy, linewidth=3.0)
    axes[0, 1].set_xlabel('Epochs ', fontsize=10)
    axes[0, 1].set_ylabel('Accuracy', fontsize=10)
    axes[0, 1].set_title('Training Accuracy Curve', fontsize=10)

    axes[1, 0].plot(range(epoch), val_loss, linewidth=3.0)
    axes[1, 0].set_xlabel('Epochs ', fontsize=10)
    axes[1, 0].set_ylabel('Loss', fontsize=10)
    axes[1, 0].set_title('Validation Loss Curve', fontsize=10)

    axes[1, 1].plot(range(epoch), val_accuracy, linewidth=3.0)
    axes[1, 1].set_xlabel('Epochs ', fontsize=10)
    axes[1, 1].set_ylabel('Accuracy', fontsize=10)
    axes[1, 1].set_title('Validation Accuracy Curve', fontsize=10)

    plt.show(block=True)



'''''
    f1_weighted = F1Score(task="multiclass", average='weighted', num_classes=len(classes))
    f1_micro = F1Score(task="multiclass", average='micro', num_classes=len(classes))
    f1_macro = F1Score(task="multiclass", average='macro', num_classes=len(classes))
    recall_weighted = Recall(task="multiclass", average='weighted', num_classes=len(classes))
    recall_micro = Recall(task="multiclass", average='micro', num_classes=len(classes))
    recall_macro = Recall(task="multiclass", average='macro', num_classes=len(classes))
    precision_micro = Precision(task="multiclass", average='micro', num_classes=len(classes))
    precision_macro = Precision(task="multiclass", average='macro', num_classes=len(classes))
    precision_weighted = Precision(task="multiclass", average='weighted', num_classes=len(classes))
    accuracy = Accuracy(task="multiclass", num_classes=len(classes))

    metrics = {
        'F1_score Micro': f1_micro(y_predicted, y_target),
        'F1_score Macro': f1_macro(y_predicted, y_target),
        'F1_score Weighted': f1_weighted(y_predicted, y_target),
        'Accuracy': accuracy(y_predicted, y_target),
        'Precision Micro': precision_micro(y_predicted, y_target),
        'Precision Macro': precision_macro(y_predicted, y_target),
        'Precision weighted': precision_weighted(y_predicted, y_target),
        'Recall Micro': recall_micro(y_predicted, y_target),
        'Recall Macro': recall_macro(y_predicted, y_target),
        'Recall weighted': recall_weighted(y_predicted, y_target)
    }
    '''



