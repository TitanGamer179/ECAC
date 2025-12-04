#Meta 2 do Projeto

import numpy as np
import os
import openfile
import graficoB

def main():
    print("\n--Requisito 1.1: Análise e Visualizção do Balanço de Classes ---\n")
    dir_path="C:\\Users\\titin\\Downloads\\vscode\\ecac\\ECAC\\Trabalho\\diretoria"
    all_data = openfile.open_all_files(dir_path)
    if all_data.size == 0:
        print("Nenhum dado foi carregado. Verifique o caminho da diretoria.")
        return
    print("Ficheiros abertos com sucesso!")
    all_activity = all_data[:, 11]
    activity_labels = all_activity[all_activity <=7]
    graficoB.generate_examples(activity_labels)
    print("Gráfico de balanço de classes gerado com sucesso!")
        