import openfile
import calculo
import graficos
import numpy as np

def main():
    #1 - Abrir os arquivos
    all_data = openfile.open_all_files('C:\\Users\\User\\Python\\ECAC\\Trabalho\\diretoria')
    print("Arquivos abertos com sucesso!")
    #2 - Realizar os cálculos
    resultados = calculo.calcular_all(all_data)
    print("Cálculos realizados com sucesso!")
    #3 - Tratamento de outliers
    data12 = []
    for i in range(len(all_data)):
        data12.append(all_data[i][11])
    #3.1 - Bloxplot
    graficos.bloxplot_activity(data12, resultados)
    #3.2 - Densidade de outliers
    densidade=calculo.calcular_densidade_outliers(all_data,2)
    print("Densidade de outliers para a atividade 2:")
    for i, d in enumerate(densidade):
       print(f"Atividade {i}: {d:.2f}%")
    #3.3 -  DETECÇÃO DE OUTLIERS - Z-SCORE
    outliers=calculo.outliers_zscore(data12,3)
    print(f"Número de outliers detectados: {np.sum(outliers)}")
    #3.4  - VISUALIZAÇÃO DE OUTLIERS Z-SCORE
    #3.5 - 3.1 VS 3.4
    #3.6 -k-Means
    #3.7 Outliers k-Means
    #4
    #4.1 TESTE DE SIGNIFICÂNCIA ESTATÍSTICA
    #4.2 EXTRAÇÃO DE FEATURES TEMPORAIS E ESPECTRAIS
    #4.3 e 4.4 PCA
    #4.5 e 4.6 Score e relieff
if __name__ == "__main__":
    main()
