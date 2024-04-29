import pandas as pd
from math import ceil
import torch
import torch.optim as optim
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader, Dataset

# Definição de uma classe de treinador personalizada que herda Seq2SeqTrainer
class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Função para calcular a perda durante o treinamento
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["decoder_input_ids"]

        # Ignorar tokens de padding na perda
        active_loss = labels.ne(model.config.pad_token_id)

        # Calcula a perda
        loss = torch.nn.CrossEntropyLoss()(logits.view(-1, logits.shape[-1]),
                                           labels.view(-1)).sum() / active_loss.sum()

        return (loss, outputs) if return_outputs else loss

# Definição de uma classe de conjunto de dados personalizada para a Task 1
class Task1Dataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.data = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Retorna um exemplo do conjunto de dados
        text = self.data.iloc[index]['texto']
        aspect = self.data.iloc[index]['aspect']
        tokens = self.tokenizer(text, padding='max_length', truncation=True, max_length=None, return_tensors='pt')
        return {
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze(),
            'aspect': aspect
        }

# Definição de uma classe de conjunto de dados personalizada para a Task 2
class Task2Dataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.data = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Retorna um exemplo do conjunto de dados
        text = self.data.iloc[index]['texto']
        aspect = self.data.iloc[index]['aspect']
        polarity = self.data.iloc[index]['polarity']
        tokens = self.tokenizer(text, padding='max_length', truncation=True, max_length=None, return_tensors='pt')
        return {
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze(),
            'aspect': aspect,
            'polarity': polarity
        }

# Função para avaliar o modelo na tarefa 1
def evaluate_model_task1(model, tokenizer, dataset):
    model.eval()
    predictions = []

    for i in range(len(dataset)):
        # Gera previsões para cada exemplo no conjunto de dados
        input_ids = dataset[i]['input_ids'].unsqueeze(0)
        attention_mask = dataset[i]['attention_mask'].unsqueeze(0)

        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=128)

        # Decodifica as previsões em texto
        predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(predicted_text)

    return predictions

# Função para avaliar o modelo na tarefa 2
def evaluate_model_task2(model, tokenizer, dataset):
    model.eval()
    predictions = []

    for i in range(len(dataset)):
        # Gera previsões para cada exemplo no conjunto de dados
        input_ids = dataset[i]['input_ids'].unsqueeze(0)
        attention_mask = dataset[i]['attention_mask'].unsqueeze(0)

        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=128)

        # Decodifica as previsões em texto
        predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Atribui a polaridade (positivo ou negativo) à previsão
        polarity = "Positivo" if predicted_text == "positive" else "Negativo"
        predictions.append(polarity)

    return predictions

# Carregamento dos dados para treinamento e teste
train = pd.read_csv('../dataset/train2024.csv', sep=';', index_col=0)
task1_test = pd.read_csv('../dataset/task1.csv', sep=';', index_col=0)
task2_test = pd.read_csv('../dataset/task2_test.csv', sep=';', index_col=0)

# Carregamento do tokenizador
tokenizer = AutoTokenizer.from_pretrained("unicamp-dl/ptt5-base-portuguese-vocab", legacy=False)

# Treinamento do modelo para a Task 1
task1_train_dataset = Task1Dataset(train, tokenizer)
model_task1 = AutoModelForSeq2SeqLM.from_pretrained("unicamp-dl/ptt5-base-portuguese-vocab")
optimizer_task1 = optim.AdamW(model_task1.parameters(), lr=5e-5)

training_args_task1 = Seq2SeqTrainingArguments(
    output_dir='./results_task1',
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs_task1',
)

trainer_task1 = CustomSeq2SeqTrainer(
    model=model_task1,
    args=training_args_task1,
    data_collator=lambda data: {'input_ids': torch.stack([f['input_ids'] for f in data]),
                                'attention_mask': torch.stack([f['attention_mask'] for f in data]),
                                'decoder_input_ids': torch.stack([f['input_ids'] for f in data])
                               },
    train_dataset=task1_train_dataset,
    tokenizer=tokenizer,
    compute_metrics=None,
    optimizers=(optimizer_task1, None),
)

trainer_task1.train()

# Avaliação do modelo na Task 1
task1_test_dataset = Task1Dataset(task1_test, tokenizer)
task1_predictions = evaluate_model_task1(model_task1, tokenizer, task1_test_dataset)

# Salva as previsões da Task 1 em um arquivo
task1_result = open("task1_result.txt", "w")
task1_result.write(str(task1_predictions))

# Treinamento do modelo para a Task 2
task2_train_dataset = Task2Dataset(train, tokenizer)
model_task2 = AutoModelForSeq2SeqLM.from_pretrained("unicamp-dl/ptt5-base-portuguese-vocab")
optimizer_task2 = optim.AdamW(model_task2.parameters(), lr=5e-5)

training_args_task2 = Seq2SeqTrainingArguments(
    output_dir='./results_task2',
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs_task2',
)

trainer_task2 = CustomSeq2SeqTrainer(
    model=model_task2,
    args=training_args_task2,
    data_collator=lambda data: {'input_ids': torch.stack([f['input_ids'] for f in data]),
                                'attention_mask': torch.stack([f['attention_mask'] for f in data]),
                                'decoder_input_ids': torch.stack([f['input_ids'] for f in data])
                               },
    train_dataset=task2_train_dataset,
    tokenizer=tokenizer,
    compute_metrics=None,
    optimizers=(optimizer_task2, None),
)

trainer_task2.train()

# Avaliação do modelo na Task 2
task2_test_dataset = Task2Dataset(task2_test, tokenizer)
task2_predictions = evaluate_model_task2(model_task2, tokenizer, task2_test_dataset)

# Salva as previsões da Task 2 em um arquivo
task2_result = open("task2_result.txt", "w")
task2_result.write(str(task2_predictions))
