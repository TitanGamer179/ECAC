import openfile
import calculo
import graficos
import numpy as np

def main():
    # 1 - Abrir os ficheiros
    all_data = openfile.open_all_files('C:\\Users\\User\\Python\\ECAC\\Trabalho\\diretoria')
    if all_data.size == 0:
        print("Nenhum dado foi carregado. Verifique o caminho da diretoria.")
        return
    print("Ficheiros abertos com sucesso!")
    
    # 2 - Calcular módulos e adicioná-los aos dados
    all_data_with_magnitudes = calculo.add_magnitude(all_data)
    print("Módulos calculados e adicionados aos dados!")

    # 3.1 - Boxplot
    print("\n--- Requisito 3.1: Boxplots por Atividade e Dispositivo ---")
    graficos.boxplot_activity(all_data_with_magnitudes)

    # 3.2 - Densidade de outliers (para o dispositivo 2 - Pulso Direito)
    print("\n--- Requisito 3.2: Densidade de Outliers (IQR) para Pulso Direito (Dispositivo 2) ---")
    densidades = calculo.calcular_densidade_outliers(all_data_with_magnitudes, 2)
    for atividade, densidade in densidades.items():
       print(f"Atividade {int(atividade)}: {densidade:.2f}%")

    # 3.3 e 3.4 - Deteção e Visualização de Outliers com Z-Score
    print("\n--- Requisitos 3.3 & 3.4: Visualização de Outliers com Z-Score ---")
    device_id_to_plot = 2
    print(f"A gerar gráficos de Z-Score para o Dispositivo {device_id_to_plot}...")
    device_data = all_data_with_magnitudes[all_data_with_magnitudes[:, 0] == device_id_to_plot]
    magnitudes_info = {12: 'Aceleração', 13: 'Giroscópio', 14: 'Magnetómetro'}
    for k in [3, 3.5, 4]:
        print(f"A calcular outliers para k={k}...")
        for col_idx, name in magnitudes_info.items():
            outliers_mask = calculo.outliers_zscore(device_data, k)
            print(f"Número de outliers detetados para {name} com k={k}: {np.sum(outliers_mask)}")
            graficos.scatter_outliers_zscore(device_data, outliers_mask, k, col_idx, name)
    print("Gráficos gerados com sucesso!")
    # 3.5 - 3.1 VS 3.4
    #3.6 -k-Means
    #3.7 Outliers k-Means
    #4
    #4.1 TESTE DE SIGNIFICÂNCIA ESTATÍSTICA
    #4.2 EXTRAÇÃO DE FEATURES TEMPORAIS E ESPECTRAIS
    #4.3 e 4.4 PCA
    #4.5 e 4.6 Score e relieff
if __name__ == "__main__":
    main()
