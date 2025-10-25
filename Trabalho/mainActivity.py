import openfile
import calculo
import graficos
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
    graficos.boxplot_all_devices(all_data_with_magnitudes)

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
        outliers_mask = calculo.outliers_zscore(device_data, k)
        graficos.scatter_3d_outliers_zscore(device_data, outliers_mask)
    print("Gráficos gerados com sucesso!")
    # 3.5 - 3.1 VS 3.4
    #3.6 -k-Means
    print("\n--- Requisito 3.6: Implementação e Visualização do k-Means ---")
    device_id = 2      
    activity_id = 4    
    n_clusters = 8     
    subset_data = all_data_with_magnitudes[(all_data_with_magnitudes[:, 0] == device_id) & (all_data_with_magnitudes[:, 11] == activity_id)]
    acc_data_3d = subset_data[:, 1:4]
    giro_data_3d = subset_data[:, 4:7]
    mag_data_3d = subset_data[:, 7:10]
    cluster_labels_acc = calculo.aplicar_kmeans(acc_data_3d, n_clusters)
    cluster_labels_giro = calculo.aplicar_kmeans(giro_data_3d, n_clusters)
    cluster_labels_mag = calculo.aplicar_kmeans(mag_data_3d, n_clusters)
    print("\nAnálise concluída.")
    # 3.7 - Deteção de Outliers com k-Means e DBSCAN
    print("\n--- Requisitos 3.7: Deteção de Outliers com k-Means ---")
    plot_title_kmeans = f'Clusters k-Means (k={n_clusters}) para Disp {device_id}, Ativ {activity_id}'
    graficos.visualizar_clusters_kmeans_3d(acc_data_3d, cluster_labels_acc, n_clusters, plot_title_kmeans)
    outliers_kmeans_mask = calculo.identificar_outliers_kmeans(cluster_labels_acc)
    plot_title_outliers_kmeans = f'Outliers k-Means para Disp {device_id}, Ativ {activity_id}'
    graficos.visualizar_outliers_3d(acc_data_3d, outliers_kmeans_mask, plot_title_outliers_kmeans)
    print(f"Número de outliers detetados com k-Means: {np.sum(outliers_kmeans_mask)}")
    print("\n--- Requisito 3.7.1: Deteção de Outliers com DBSCAN ---")
    outliers_dbscan_mask = calculo.aplicar_dbscan(acc_data_3d, eps=0.5, min_samples=20)
    plot_title_outliers_dbscan = f'Outliers DBSCAN para Disp {device_id}, Ativ {activity_id}'
    graficos.visualizar_outliers_3d(acc_data_3d, outliers_dbscan_mask, plot_title_outliers_dbscan)
    print(f"Número de outliers detetados com DBSCAN: {np.sum(outliers_dbscan_mask)}")
    print("\nAnálise concluída.")
    #4
    #4.1 TESTE DE SIGNIFICÂNCIA ESTATÍSTICA
    #4.2 EXTRAÇÃO DE FEATURES TEMPORAIS E ESPECTRAIS
    #4.3 e 4.4 PCA
    #4.5 e 4.6 Score e relieff
if __name__ == "__main__":
    main()
