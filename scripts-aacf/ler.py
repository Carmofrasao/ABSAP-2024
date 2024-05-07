import re
import pandas as pd

# Função para encontrar a frase que contém a palavra a partir do índice
def encontrar_frase_com_indice(frase, indice):
    # Encontrar o início da frase após a última pontuação
    inicio_frase = 0
    for i in range(indice, 0, -1):
        if frase[i] in [".", ";", "!", "?"]:
            inicio_frase = i + 1
            break

    # Encontrar o final da frase antes da próxima pontuação após o índice
    final_frase = len(frase)
    for i in range(indice, len(frase)):
        if frase[i] in [".", ";", "!", "?"]:
            final_frase = i
            break

    return frase[inicio_frase:final_frase]

data = pd.read_csv('dataset/task2_test.csv', sep=';', index_col=0)

for i in range(len(data)):
    frase = data.iloc[i]["texto"]
    indice_inicial = data.iloc[i]["start_position"]
    frase_completa = encontrar_frase_com_indice(frase, indice_inicial)
    print(frase_completa)
    print()
