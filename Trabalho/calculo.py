import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

# Função para calcular o tratamento de outliers
def add_magnitude(data):
    mod_acc = np.sqrt(data[:, 1]**2 + data[:, 2]**2 + data[:, 3]**2)
    mod_gyro = np.sqrt(data[:, 4]**2 + data[:, 5]**2 + data[:, 6]**2)
    mod_mag = np.sqrt(data[:, 7]**2 + data[:, 8]**2 + data[:, 9]**2)
    return np.c_[data, mod_acc, mod_gyro, mod_mag]

# Função para detetar outliers usando o método do IQR
def get_iqr_outliers(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr=q3 - q1
    lim_inferior = q1 - 1.5 * iqr
    lim_superior = q3 + 1.5 * iqr
    return (data<lim_inferior)|(data>lim_superior)

# Função para calcular a densidade de outliers por atividade
def calcular_densidade_outliers(data,id):
    dados_id=data[data[:,0]==id]
    atividades=sorted(np.unique(dados_id[:,11]))
    densidade={}
    for a in atividades:
        dados_ativ=dados_id[dados_id[:,11]==a]
        n_total=len(dados_ativ)
        mod_acc=dados_ativ[:,12]
        mod_gyro=dados_ativ[:,13]
        mod_mag=dados_ativ[:,14]
        out_acc=get_iqr_outliers(mod_acc)
        out_gyro=get_iqr_outliers(mod_gyro)
        out_mag=get_iqr_outliers(mod_mag)
        n_outliers=np.sum(out_acc|out_gyro|out_mag)
        densidade[a]=(n_outliers/n_total)*100
    return densidade

# Função para detectar outliers usando Z-score
def outliers_zscore(data, threshold):
    outliers=np.zeros(len(data), dtype=bool)
    atividades=np.unique(data[:,11])
    for a in atividades:
        indices_ativ=np.where(data[:,11]==a)[0]
        dados_ativ=data[indices_ativ]
        for col_idx in [12,13,14]:  # Colunas de magnitude
            col_data=dados_ativ[:,col_idx]
            mean = np.mean(col_data)
            std_dev = np.std(col_data)
            z_scores = (col_data - mean) / std_dev
            is_outlier = np.abs(z_scores) > threshold
            outliers[indices_ativ[is_outlier]] = True
    return outliers

def aplicar_kmeans(data_3d, n_clusters):
    print(f"A aplicar o k-means com n={n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(data_3d)
    print("K-means aplicado com sucesso.")
    return labels

def identificar_outliers_kmeans(labels, min_size_percent=1.0):
    print("A identificar outliers com base nos clusters do k-means...")
    unique_labels, counts = np.unique(labels, return_counts=True)
    min_cluster_size = len(labels) * (min_size_percent / 100.0)
    outlier_clusters = unique_labels[counts < min_cluster_size]
    is_outlier_mask = np.isin(labels, outlier_clusters)
    print(f"Identificados {len(outlier_clusters)} clusters de outliers (com menos de {min_cluster_size:.0f} pontos).")
    return is_outlier_mask

def aplicar_dbscan(data_3d, eps=0.5, min_samples=15):
    print(f"A aplicar o DBSCAN com eps={eps} e min_samples={min_samples}...")
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(data_3d)
    is_outlier_mask = (labels == -1)
    print("DBSCAN aplicado com sucesso.")
    return is_outlier_mask