import pandas as pd
import numpy as np
from preprocecss import *
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def metrics(y_preds, y_true):
  auc = roc_auc_score(y_true, y_preds)
  accuracy = accuracy_score(y_true, y_preds>0.5)
  f1 = f1_score(y_true, y_preds>0.5, average='macro')

  return accuracy, f1, auc
  
def eval(val_loader, model, device, num_labels=7):
    # Evaluate model
    model.eval()
    y_val = []
    val_preds = None
    for (input_ids, attention_mask, y_batch) in val_loader:
        y_pred = model(input_ids.to(device), attention_mask=attention_mask.to(device), token_type_ids=None)
        y_pred = y_pred.squeeze().detach().cpu().numpy()
        
        val_preds = np.atleast_1d(y_pred) if val_preds is None else np.concatenate(
            [val_preds, np.atleast_1d(y_pred)])
        y_val.extend(y_batch.tolist())
    
    y_val = np.array(y_val)
    accuracy_score = 0
    f1_score = 0
    auc_score = 0
    for i in range(num_labels):
      acc, f1, auc = metrics(val_preds[:,i], y_val[:,i])
      accuracy_score += acc 
      f1_score += f1
      auc_score += auc 

    print(f"\n----- F1 score = = {f1_score/num_labels:.4f}")
    print("Accuracy: ", accuracy_score/num_labels)
    print("AUC_core:", auc_score/num_labels)

def predict(sentences, model, tokenizer, max_seq_len, device, vectorizer, max_len_tfidf):
    input_ids, attention_masks = convert_samples_to_ids(sentences, tokenizer, max_seq_len, vectorizer, max_len_tfidf)
    model.eval()

    y_pred = model(input_ids.to(device), attention_masks.to(device), token_type_ids=None )
    y_pred = y_pred.squeeze().detach().cpu().numpy()

    return y_pred