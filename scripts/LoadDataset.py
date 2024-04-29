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

# Definição de uma classe de conjunto de dados personalizada
class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.data = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Retorna um exemplo do conjunto de dados
        text = self.data.iloc[index]['texto']
        tokens = self.tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        return {
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze(),
        }

# Função para avaliar o modelo nos conjuntos de teste
def evaluate_model(model, tokenizer, dataset):
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

# Carregamento e divisão dos dados de treinamento e teste
trainC = pd.read_csv('../dataset/train2024.csv', sep=';', index_col=0)
trainC = trainC.sample(frac=1)  # Embaralha os dados

# Definição de proporções para validação e treinamento
VP = ceil((0.02 * (len(trainC) + 1)) / 0.1)
OP = len(trainC) + 1 - VP

# Divisão dos dados em conjuntos de treinamento e teste
train = trainC.iloc[:OP,]
test = trainC.iloc[OP:,]

# Carregamento do tokenizador e criação do conjunto de dados de treinamento
tokenizer = AutoTokenizer.from_pretrained("unicamp-dl/ptt5-base-portuguese-vocab", legacy=False)
train_dataset = CustomDataset(train, tokenizer)

# Carregamento do modelo pré-treinado
model = AutoModelForSeq2SeqLM.from_pretrained("unicamp-dl/ptt5-base-portuguese-vocab")

# Definição do otimizador para o treinamento do modelo
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

# Definição dos argumentos de treinamento
training_args = Seq2SeqTrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs',
)

# Configuração do treinador com os parâmetros definidos
trainer = CustomSeq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=lambda data: {'input_ids': torch.stack([f['input_ids'] for f in data]),
                                'attention_mask': torch.stack([f['attention_mask'] for f in data]),
                                'decoder_input_ids': torch.stack([f['input_ids'] for f in data])
                               },
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    compute_metrics=None,
    optimizers=(optimizer, None),
)

# Treinamento do modelo
trainer.train()

# Avaliação do modelo no conjunto de teste
test_dataset = CustomDataset(test, tokenizer)
test_predictions = evaluate_model(model, tokenizer, test_dataset)

# Salva as previsões do conjunto de teste em um arquivo
test_result = open("test_result.txt", "w")
test_result.write(str(test_predictions))

# Carrega os dados para a Task 1
task1 = pd.read_csv('../dataset/task1_test.csv', sep=';', index_col=0)

# Avaliação do modelo na Task 1
task1_dataset = CustomDataset(task1, tokenizer)
task1_predictions = evaluate_model(model, tokenizer, task1_dataset)

# Salva as previsões da Task 1 em um arquivo
task1_result = open("task1_result.txt", "w")
task1_result.write(str(task1_predictions))

# Carrega os dados para a Task 2
task2 = pd.read_csv('../dataset/task2_test.csv', sep=';', index_col=0)

# Avaliação do modelo na Task 2
task2_dataset = CustomDataset(task2, tokenizer)
task2_predictions = evaluate_model(model, tokenizer, task2_dataset)

# Salva as previsões da Task 2 em um arquivo
task2_result = open("task2_result.txt", "w")
task2_result.write(str(task2_predictions))

# Exibe alguns exemplos de previsões para as tarefas
print("Exemplos de previsões para Task1:")
for i in range(3):
    print(f"Texto Original: {task1.iloc[i]['texto']}")
    print(f"Previsão: {task1_predictions[i]}")
    print("------")

print("\nExemplos de previsões para Task2:")
for i in range(3):
    print(f"Texto Original: {task2.iloc[i]['texto']}")
    print(f"Previsão: {task2_predictions[i]}")
    print("------")
