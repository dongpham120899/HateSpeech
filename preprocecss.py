import numpy as np
import pandas as pd 
import torch


def convert_samples_to_ids(texts, tokenizer, max_seq_length, vectorizer, max_len_tf_idf,labels=None):
    input_ids, attention_masks = [], []

    for text in texts:
        inputs = tokenizer.encode_plus(text, padding='max_length', max_length=max_seq_length, truncation=True)
        input_ids.append(inputs['input_ids'])
        masks = inputs['attention_mask']
        attention_masks.append(masks)

    # TF-IDF
    tf_idf = vectorizer.transform(texts)
    tf_idf = abs(tf_idf*1000)

    input_ids = np.concatenate((input_ids, tf_idf), axis=1)
    attention_masks = np.concatenate((attention_masks, np.zeros((len(texts), max_len_tf_idf))), axis=1)

    if labels is not None:
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_masks, dtype=torch.long), torch.tensor(
            labels, dtype=torch.long)
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_masks, dtype=torch.long)

def get_dataloader(sentences, tokenizer, labels, batch_size, max_seq_length, max_len_tf_idf, vectorizer):
    input_ids, masks, labels = convert_samples_to_ids(texts=sentences, tokenizer=tokenizer, max_seq_length=max_seq_length, max_len_tf_idf=max_len_tf_idf, vectorizer=vectorizer, labels=labels)
    dataset = torch.utils.data.TensorDataset(input_ids, masks, labels)
    sampler = torch.utils.data.RandomSampler(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    return dataloader
