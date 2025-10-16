import csv
import os
import numpy as np

# Função para abrir um arquivo CSV e retornar os dados como uma lista de listas
def open_file(file_path):
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        data = list(reader)
    return data

# Função para abrir todos os arquivos CSV em uma diretoria e retornar os dados combinados
def open_all_files(directory):
    all_data = []
    contagem_pastas = len([name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))])
    for i in range(contagem_pastas):
        quant_itens=len(os.listdir(os.path.join(directory, f"part{i}")))
        for j in range(1, quant_itens + 1):
            data = open_file(os.path.join(directory, f"part{i}/part{i}dev{j}.csv"))
            data_float=[[float(valor) for valor in linha] for linha in data]
            all_data.extend(data_float)
    return np.array(all_data)