#Meta 2 do Projeto

import numpy as np
import os
import openfile
import graficoB
import calculo

def main():
    print("\n--Requisito 1.1: Análise e Visualizção do Balanço de Classes ---\n")
    dir_=os.path.dirname(os.path.abspath(__file__))
    dir_path=os.path.join(dir_,'diretoria')
    all_data = openfile.open_all_files(dir_path)
    if all_data.size == 0:
        print("Nenhum dado foi carregado. Verifique o caminho da diretoria.")
        return
    print("Ficheiros abertos com sucesso!")
    all_data=calculo.add_magnitude(all_data);
    all_data=all_data[all_data[:, -1] <= 7] 
    print("Dados filtrados com sucesso!")
    feats,lables,parts=calculo.extrair_features(all_data)
    segs_raw, labels_seg, parts_seg = calculo.segmentation(all_data)

if __name__ == "__main__":
    main()