from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


result = {}
def measureMulti (test_labels,predictions):
    precision = precision_score(test_labels, predictions,
                                average='micro')
    recall = recall_score(test_labels, predictions,
                          average='micro')
    f1 = f1_score(test_labels, predictions, average='micro')

    print("Micro-average quality numbers")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}"
          .format(precision, recall, f1))

    precision = precision_score(test_labels, predictions,
                                average='macro')
    recall = recall_score(test_labels, predictions,
                          average='macro')
    f1 = f1_score(test_labels, predictions, average='macro')

    print("Macro-average quality numbers")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}"
          .format(precision, recall, f1))
    result["precision"] = precision
    result["recall"] = recall
    result["f-measure"] = f1
    return result

def measureNormal (test_labels,predictions):
    precision = precision_score(test_labels, predictions,
                                average='micro')
    recall = recall_score(test_labels, predictions,
                          average='micro')
    f1 = f1_score(test_labels, predictions, average='micro')

    #auc = roc_auc_score(test_labels, predictions)

    print("Micro-average quality numbers")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}"
          .format(precision, recall, f1))
    result["precision"] = precision
    result["recall"] = recall
    result["f-measure"] = f1
    return result
