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

# Função para criar boxplots para todos os dispositivos
def boxplot_all_devices(all_data):
    magnitudes_info = {12: 'Aceleração', 13: 'Giroscópio', 14: 'Magnetómetro'}
    activity_labels = sorted(np.unique(all_data[:, 11]))
    fig, axs = plt.subplots(1, 3, figsize=(20, 6), sharey=False)
    fig.suptitle('Boxplots para Todos os Dispositivos', fontsize=16)
    for idx, (col, title) in enumerate(magnitudes_info.items()):
        data_to_plot = [all_data[all_data[:, 11] == a, col] for a in activity_labels]
        axs[idx].boxplot(data_to_plot)
        axs[idx].set_title(f'Módulo de {title}')
        axs[idx].set_xlabel('Atividade')
        axs[idx].set_ylabel('Módulo')
        axs[idx].grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
def scatter_3d_outliers_zscore(all_data, outliers_mask):
    from mpl_toolkits.mplot3d import Axes3D  # Import necessário para gráficos 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    inlier_data = all_data[~outliers_mask]
    outlier_data = all_data[outliers_mask]
    ax.scatter(inlier_data[:, 12], inlier_data[:, 13], inlier_data[:, 14], color='blue', label='Inliers', alpha=0.5, s=10)
    ax.scatter(outlier_data[:, 12], outlier_data[:, 13], outlier_data[:, 14], color='red', label='Outliers', marker='x')
    ax.set_title('Deteção de Outliers com Z-Score - Gráfico 3D')
    ax.set_xlabel('Módulo de Aceleração')
    ax.set_ylabel('Módulo de Giroscópio')
    ax.set_zlabel('Módulo de Magnetómetro')
    ax.legend()
    plt.show()
    
def visualizar_outliers_3d(data_3d, outliers_mask, title):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    inliers = data_3d[~outliers_mask]
    outliers = data_3d[outliers_mask]
    ax.scatter(inliers[:, 0], inliers[:, 1], inliers[:, 2],c='blue', label='Inliers', alpha=0.4, s=15)
    ax.scatter(outliers[:, 0], outliers[:, 1], outliers[:, 2],c='red', label='Outliers', marker='x', s=50)
    ax.set_title(title)
    ax.set_xlabel('Acelerómetro Eixo X')
    ax.set_ylabel('Acelerómetro Eixo Y')
    ax.set_zlabel('Acelerómetro Eixo Z')
    ax.legend()
    plt.show()
    
def visualizar_clusters_kmeans_3d(data_3d, labels, n_clusters, title):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2],
                         c=labels, cmap='viridis', s=15, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel('Acelerómetro Eixo X'); ax.set_ylabel('Acelerómetro Eixo Y')
    ax.set_zlabel('Acelerómetro Eixo Z')
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    plt.show()