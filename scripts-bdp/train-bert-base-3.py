#!/usr/bin/python3 -u

import sys

class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)

from datasets import Dataset, load_dataset, load_metric
from sklearn.metrics import f1_score
from transformers import AdamW, get_scheduler, AutoTokenizer, DataCollatorWithPadding, BertForPreTraining, AutoModelForSequenceClassification
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import pandas as pd
from os import sys
from sklearn import preprocessing
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from evaluate import load

def train(model,iterator,epoca,optimizer):
    print(f'Iniciando treinamento da epoca {epoca}')
    epoch_loss = 0.0
    metric = load("accuracy")
    metric2 = load("f1")

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        labels = batch["target"].to(model.device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        logits = outputs[1]

        loss.mean().backward()
        optimizer.step()

        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=labels)
        metric2.add_batch(predictions=predictions, references=labels)

        epoch_loss += loss.mean().item()

    accuracy = metric.compute()["accuracy"]
    f1 = metric2.compute(average="weighted")["f1"]
    return epoch_loss / len(iterator), accuracy, f1

def analize(model,iterator,epoca):
    print(f'Iniciando avaliação da epoca {epoca}')
    epoch_loss = 0.0
    metric = load("accuracy")
    metric2 = load("f1")

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["target"].to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            logits = outputs[1]

            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=labels)
            metric2.add_batch(predictions=predictions, references=labels)

            epoch_loss += loss.mean().item()

    accuracy = metric.compute()["accuracy"]
    f1 = metric2.compute(average="weighted")["f1"]
    return epoch_loss / len(iterator), accuracy, f1

def test(model,dataloader, tokenizer):
    print(f'Iniciando teste')
    aspects = []
    idss = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            idss+=[int(i) for i in batch["id"]]
            decoded_aspects = label_encoder.inverse_transform(predictions.cpu().numpy())
            aspects.append(decoded_aspects)
    mapp=dict(zip(idss,aspects))
    return mapp

def get_aspect_phrase(review, aspect_start, aspect_end):
    padded_review = "." + review + "."
    start = aspect_start
    end = aspect_end
    while padded_review[start] != '.' or padded_review[end] != '.':
        if padded_review[start] != '.':
            start -= 1
        if padded_review[end] != '.':
            end += 1
    return padded_review[start+1:end+1]

def preprocess_review(row):
    row['texto'] = get_aspect_phrase(row['texto'], int(row['start_position']), int(row['end_position']))
    row['aspect'] = row['aspect']
    return row

tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', model_max_length = 512, padding_side='left')

# train_data_filepath = 'dataset-bert/train-sample2-fixed.csv'
train_data_filepath = 'dataset-bert/train-sample.csv'
test_data_filepath = 'dataset-bert/test-sample.csv'
final_eval_filepath = 'dataset-bert/task1_test.csv'

raw_datasets_train = load_dataset('csv', data_files=train_data_filepath, delimiter=';')
preprocessed_datasets_train = raw_datasets_train.map(preprocess_review)
tokenized_datasets_train = preprocessed_datasets_train.map(lambda x: tokenizer(x['texto'], truncation=True, padding='max_length', max_length=50), batched=True)
tokenized_datasets_train = tokenized_datasets_train.rename_column('aspect', 'target')
tokenized_datasets_train = tokenized_datasets_train.remove_columns(['id', 'texto', 'polarity', 'start_position', 'end_position'])
aspectos_train = tokenized_datasets_train['train']['target']

raw_datasets_test = load_dataset('csv', data_files=test_data_filepath, delimiter=';')
preprocessed_datasets_test = raw_datasets_test.map(preprocess_review)
tokenized_datasets_test = preprocessed_datasets_test.map(lambda x: tokenizer(x['texto'], truncation=True, padding='max_length', max_length=50), batched=True)
tokenized_datasets_test = tokenized_datasets_test.rename_column('aspect', 'target')
tokenized_datasets_test = tokenized_datasets_test.remove_columns(['id', 'texto', 'polarity', 'start_position', 'end_position'])
aspectos_test = tokenized_datasets_test['train']['target']

# Inicializar o codificador de rótulos
label_encoder = preprocessing.LabelEncoder()

# Ajustar o codificador de rótulos aos aspectos e transformar em valores numéricos
target_encoded = label_encoder.fit_transform(aspectos_train + aspectos_test)
d = dict()
for i, aspecto in enumerate(aspectos_train + aspectos_test):
    if aspecto in d: assert(target_encoded[i] == d[aspecto])
    d[aspecto] = target_encoded[i]

# Função para transformar os rótulos com padding
def encode_labels_with_padding(example, max_length):
    aspect_tensor = torch.tensor([d[example['target']]], dtype=torch.long)
    padded_tensor = torch.nn.functional.pad(aspect_tensor, (0, max_length - len(aspect_tensor)))
    example['target'] = padded_tensor
    return example

# Aplicar a função de transformação com padding aos dados
max_length = 1  # Defina o comprimento máximo desejado
tokenized_datasets_train = tokenized_datasets_train.map(lambda example: encode_labels_with_padding(example, max_length))

tokenized_datasets_train.set_format("torch")

# Aplicar a função de transformação com padding aos dados
max_length = 1  # Defina o comprimento máximo desejado
tokenized_datasets_test = tokenized_datasets_test.map(lambda example: encode_labels_with_padding(example, max_length))

tokenized_datasets_test.set_format("torch")

raw_datasets_final = load_dataset('csv', data_files=final_eval_filepath, delimiter=';')
preprocessed_datasets_final = raw_datasets_final
tokenized_datasets_final = preprocessed_datasets_final.map(lambda x: tokenizer(x['texto'], truncation=True, padding='max_length', max_length=50), batched=True)
tokenized_datasets_final = tokenized_datasets_final.remove_columns(['texto'])
tokenized_datasets_final.set_format("torch")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

batch_size = 1 

train_dataloader = DataLoader(
    tokenized_datasets_train["train"], shuffle=True, batch_size=batch_size, collate_fn=data_collator
)
test_dataloader = DataLoader(
    tokenized_datasets_test["train"], batch_size=batch_size, collate_fn=data_collator
)
final_dataloader = DataLoader(
    tokenized_datasets_final["train"], batch_size=batch_size, collate_fn=data_collator
)
for data in train_dataloader:
  for d in data['input_ids']:
    print(d)
    decoded = label_encoder.inverse_transform(d.cpu().numpy())
    print(decoded)
  print(data['target'])
  exit(1)
# epoch_number = 10
epoch_number = 1

num_labels = len(d)

model = AutoModelForSequenceClassification.from_pretrained(
    "neuralmind/bert-base-portuguese-cased", 
    num_labels=num_labels,
    hidden_dropout_prob=0.3,  # Adicione dropout
    attention_probs_dropout_prob=0.3  # Adicione dropout
)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=epoch_number * len(train_dataloader),)

model.to('cuda' if torch.cuda.is_available() else 'cpu')

for epoch in range(1, epoch_number + 1):
    print(f"\t Epoch: {epoch}", flush=True)
    train_loss, train_acc, train_f1 = train(model, train_dataloader, epoch, optimizer)
    valid_loss, valid_acc, valid_f1 = analize(model, test_dataloader, epoch)

    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f} | Train f1: {train_f1*100:.2f}%', flush=True)
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f} |  val. f1: {valid_f1*100:.2f}%', flush=True)
    print()

aspects = test(model, final_dataloader, tokenizer)
print("\"input id number\";\"list of aspects\"")
for key,value in aspects.items():
    print(f'{key};\"{value}\"', flush=True)
