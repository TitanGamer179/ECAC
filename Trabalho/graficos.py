import matplotlib.pyplot as plt
import numpy as np

# Função para criar boxplots das atividades
def boxplot_activity(all_data):
    device_ids=sorted(np.unique(all_data[:,0]))
    magnitudes_info = {12: 'Aceleração',13: 'Giroscópio',14: 'Magnetómetro'}
    for dev_id in device_ids:
        fig,axs=plt.subplots(1,3, figsize=(20, 6), sharey=False)
        fig.suptitle(f'Boxplots for Device {int(dev_id)}', fontsize=16)
        device_data=all_data[all_data[:,0]==dev_id]
        activity_labels=sorted(np.unique(device_data[:,11]))
        for idx, (col, title) in enumerate(magnitudes_info.items()):
            data_to_plot=[device_data[device_data[:,11]==a,col] for a in activity_labels]
            axs[idx].boxplot(data_to_plot)
            axs[idx].set_title(f'Módulo de {title}')
            axs[idx].set_xlabel('Atividade')
            axs[idx].set_ylabel('Módulo')
            axs[idx].grid(True)
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

# Função para criar gráficos de dispersão dos outliers detectados pelo Z-score
def scatter_outliers_zscore(device_data, outliers,k,magnitude_col,magnitude_name):
    plt.figure(figsize=(12, 7))
    inlier_data = device_data[~outliers]
    outlier_data = device_data[outliers]
    plt.scatter(inlier_data[:, 11], inlier_data[:, magnitude_col], color='blue', label='Inliers', alpha=0.5, s=10)
    plt.scatter(outlier_data[:, 11], outlier_data[:, magnitude_col], color='red', label='Outliers', marker='x')
    plt.title(f'Deteção de Outliers com Z-Score (k={k}) - Módulo do {magnitude_name}')
    plt.xlabel('Atividade')
    plt.ylabel('Valor do Módulo')
    plt.xticks(sorted(np.unique(device_data[:, 11]).astype(int)))
    plt.legend()
    plt.grid(True)
    plt.show()