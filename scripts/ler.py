"""
import pandas as pd

data = pd.read_csv('../dataset/task2_test.csv', sep=';', index_col=0)

for i in range(len(data)):
    frase = data.iloc[i]["texto"]
    
    palavras = frase.split()
    
    # Índice do caractere a partir do qual queremos extrair as palavras
    indice_inicial = data.iloc[i]["start_position"]

    # Encontrando as palavras a partir do sétimo caractere
    palavras_a_partir_do_index = []
    for palavra in palavras:
        indice_palavra = frase.find(palavra)
        if indice_palavra <= indice_inicial:
            if indice_palavra + len(palavra) >= indice_inicial:
                palavras_a_partir_do_index.append(palavra)
        else:
            palavras_a_partir_do_index.append(palavra)

    # Juntando as palavras encontradas de volta em uma frase
    frase_extraida = " ".join(palavras_a_partir_do_index)

    print(f'{frase_extraida} --- {data.iloc[i]["aspect"]}')
    print()
"""
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

data = pd.read_csv('../dataset/task2_test.csv', sep=';', index_col=0)

for i in range(len(data)):
    frase = data.iloc[i]["texto"]
    indice_inicial = data.iloc[i]["start_position"]
    frase_completa = encontrar_frase_com_indice(frase, indice_inicial)
    print(frase_completa)
    print()
