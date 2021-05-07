import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoConfig, AdamW, get_linear_schedule_with_warmup, get_constant_schedule
import torch
import label_attention_layer as label_attention
import torch.nn as nn
from models import *
from sklearn.model_selection import train_test_split
from utils import eval, sigmoid, predict
from preprocecss import convert_samples_to_ids, get_dataloader
from tqdm.notebook import tqdm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import argparse


# device
if torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')

def train_model(train_dataloader, val_dataloader, model, EPOCHS, BATCH_SIZE, lr, ACCUMULATION_STEPS):
    ## Optimization
    num_train_optimization_steps = int(EPOCHS * len(train_dataloader) / BATCH_SIZE / ACCUMULATION_STEPS)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(np in n for np in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(np in n for np in no_decay)], 'weight_decay': 0.01}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100,
                                            num_training_steps=num_train_optimization_steps)
    scheduler0 = get_constant_schedule(optimizer)
    

    frozen = True
    # Training
    for epoch in (range(EPOCHS+1)):
        print("\n--------Start training on  Epoch %d/%d" %(epoch, EPOCHS))
        avg_loss = 0 
        avg_accuracy = 0

        model.train()
        for i, (input_ids, attention_mask, label_batch) in (enumerate(train_dataloader)):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            label_batch = label_batch.to(device)

            optimizer.zero_grad()
            y_preds = model(input_ids, attention_mask, None)
            loss = torch.nn.functional.binary_cross_entropy(y_preds.to(device),
                                                                label_batch.float().to(device))
            
            loss = loss.mean()
            loss.backward()
            optimizer.step()

            lossf = loss.item()
            avg_loss += loss.item() / len(train_dataloader)

        print("Loss training:", avg_loss)

        roc = eval(val_dataloader, model, device)

    return model

def load_datasets(path_train, path_test):
  train = pd.read_csv(path_train)
  test = pd.read_csv(path_test)
  train = train.dropna()
  test = test.dropna()

  return train, test


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="PhoBert CNN_HateSpeech Detection")
  parser.add_argument("-batch_size", type=int, default=16)
  parser.add_argument("-epochs", type=int, default=10)
  parser.add_argument("-learning_rate", type=float, default=1e-5)
  parser.add_argument("-max_seq_len", type=int, default=256)
  parser.add_argument("-bacth-size", type=int, default=16)
  parser.add_argument("-kernel_sizes", type=int, default=[3,5,7])
  parser.add_argument("-max_len_tfidf", type=int, default=0, help="If use TF.IDF, set max_len_tf.idf apart 0")
  parser.add_argument("-drop_out", type=float, default=0.1)
  parser.add_argument("-accumulation_steps", type=float, default=1)
  parser.add_argument("-num_labels", type=int, default=7)
  parser.add_argument("-label_attention", type=bool, default=False, help="If label_attention is True then using Label Attention")
  args = parser.parse_args()

  model_name = 'vinai/phobert-base'
  labels = ['Toxicity','Obscence','Threat','Identity attack - Insult','Sexual explicit','Sedition – Politics','Spam']
  comments = "vncore_comments"
  # Loading datasets
  train, test = load_datasets("Datasets/train_v6.csv","Datasets/test_v6.csv")
  train_sents = train[comments].values
  valid_sents = test[comments].values
  train_labels = train[labels].values
  valid_labels = test[labels].values

  # TF.IDF + SVD
  clf = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,3))),
    ('svd', TruncatedSVD(n_components=args.max_len_tfidf, random_state=42))
  ])
  clf.fit(np.hstack((train_sents, valid_sents)))
  ## Tokenizer PhoBERT
  tokenizer = AutoTokenizer.from_pretrained(model_name, use_fact=False)

  # Loading config model
  config = AutoConfig.from_pretrained(model_name)
  config.output_hidden_states = True
  config.num_labels = args.num_labels
  # Loading model
  model = BertCNN.from_pretrained(model_name,
                                config=config,
                                MODEL_NAME = model_name,
                                drop_out=args.drop_out,
                                lb_availible=args.label_attention,
                                max_len=args.max_seq_len,
                                max_len_tfidf=args.max_len_tfidf,
                                kernel_sizes=args.kernel_sizes)
  model.to(device)

  #Dataloader
  train_loader = get_dataloader(train_sents, tokenizer, train_labels, args.batch_size, args.max_seq_len, args.max_len_tfidf, clf)
  valid_loader = get_dataloader(valid_sents, tokenizer, valid_labels, args.batch_size, args.max_seq_len, args.max_len_tfidf, clf)

  ## Training 
  model = train_model(train_loader, valid_loader, model, args.epochs, args.batch_size, args.learning_rate, args.accumulation_steps)