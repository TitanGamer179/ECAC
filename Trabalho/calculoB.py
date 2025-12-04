import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.signal import resample
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
import random
import calculo

# Requisito 1.1: Análise do Balanço de Classes
def check_balance(labels):
    unique, counts = np.unique(labels, return_counts=True)
    balance_dict = dict(zip(unique, counts))
    print ("Distribuição das classes:", balance_dict)
    if max(counts) / min(counts) > 1.5:
        print("Os dados estão desbalanceados.")
    else:
        print("Os dados estão balanceados.")
    

# Requisito 1.2: Implementação do SMOTE
def custom_smote(features, labels,target_class, N=100, k=5):
    dados=features[labels == target_class]
    if len(dados)<k+1:return np.array([])
    neigh = NearestNeighbors(n_neighbors=k+1)
    neigh.fit(dados)
    synthetic_samples = []
    for _ in range(N):
        indeces=random.randint(0, len(dados) - 1)
        x_i = dados[indeces]
        neighbors = neigh.kneighbors([x_i], return_distance=False)[0]
        neighbor=dados[neighbors[random.randint(1, k)]]
        diff = neighbor - x_i
        new_dados= x_i + (diff * random.random())
        synthetic_samples.append(new_dados)
    return np.array(synthetic_samples)

def apply_smote(features, labels):
    unique, counts = np.unique(labels, return_counts=True)
    balance_dict = dict(zip(unique, counts))
    max_count = max(counts)
    synthetic_features = []
    synthetic_labels = []
    for cls in unique:
        n_samples = balance_dict[cls]
        N = max_count - n_samples
        if N > 0:
            synthetic_data = custom_smote(features, labels, cls, N=N, k=5)
            if synthetic_data.size > 0:
                synthetic_features.append(synthetic_data)
                synthetic_labels.extend([cls] * synthetic_data.shape[0])
    if synthetic_features:
        synthetic_features = np.vstack(synthetic_features)
        augmented_features = np.vstack((features, synthetic_features))
        augmented_labels = np.hstack((labels, np.array(synthetic_labels)))
        return augmented_features, augmented_labels
    else:
        return features, labels

# Requesito 2.1: Extração de Features de Embedding
def extract_embedding_features(features,target_fs=30,n_components=64):
    embedded_features = []
    target_sample=int(5*target_fs)
    for feat in features:
        acc_dados=feat[:,:3]
        if len(acc_dados)>0:
            seg_resized=resample(acc_dados,target_sample)
        else:
            seg_resized=np.zeros((target_sample,3))
        embedded_features.append(seg_resized.flatten())
    matrix=np.array(embedded_features)
    scaler = StandardScaler()
    matrix_scaled = scaler.fit_transform(matrix)
    n_components = min(n_components, matrix_scaled.shape[1], matrix_scaled.shape[0])
    pca = PCA(n_components=n_components)
    embedding = pca.fit_transform(matrix_scaled)
    return embedding

# Requisito 3.1: TVT Split 60-20-20 por Participante
def tvt_split(features, labels, parts, train_ratio=0.6, val_ratio=0.2):
    test=StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_val_idx, test_idx = next(test.split(features, labels))
    x_train_val=features[train_val_idx]
    y_train_val=labels[train_val_idx]
    original_idx_train_val=train_val_idx
    val=StratifiedShuffleSplit(n_splits=1, test_size=val_ratio/(train_ratio + val_ratio), random_state=42)
    train_idx, val_idx = next(val.split(x_train_val, y_train_val))
    final_train_idx=original_idx_train_val[train_idx]
    final_val_idx=original_idx_train_val[val_idx]
    n_total=len(features)
    print(f"Split Within-Subject realizado:")
    print(f"  Treino: {len(final_train_idx)} ({len(final_train_idx)/n_total:.1%}) -> Esperado 60%")
    print(f"  Valid:  {len(final_val_idx)} ({len(final_val_idx)/n_total:.1%}) -> Esperado 20%")
    print(f"  Teste:  {len(test_idx)} ({len(test_idx)/n_total:.1%}) -> Esperado 20%")
    return final_train_idx, final_val_idx, test_idx

# Requisito 3.2: TVT Split por Participante
def split_between_subjects(features, labels, parts):
    unique_parts = np.unique(parts)
    if(len(unique_parts)<15):
        raise ValueError("Número insuficiente de participantes para divisão entre sujeitos.")
    np.random.seed(42)
    shuffled_parts = np.random.permutation(unique_parts)
    train_p_ids=shuffled_parts[:9]
    val_p_ids=shuffled_parts[9:12]
    test_p_ids=shuffled_parts[12:]
    print(f"Participantes atribuídos:")
    print(f"  Treino: {train_p_ids}")
    print(f"  Validação: {val_p_ids}")
    print(f"  Teste: {test_p_ids}")
    train_idx = np.where(np.isin(parts, train_p_ids))[0]
    val_idx = np.where(np.isin(parts, val_p_ids))[0]
    test_idx = np.where(np.isin(parts, test_p_ids))[0]
    return train_idx, val_idx, test_idx

# Requisito 3.3: Combinação de Estratégias de Divisão
def combined_split(x_train, y_train, x_val,x_test, method):
    scaler= StandardScaler()
    x_train_norm=scaler.fit_transform(x_train)
    x_val_norm=scaler.transform(x_val)
    x_test_norm=scaler.transform(x_test)
    selector = None
    if method == "all":
        return x_train_norm, x_val_norm, x_test_norm,scaler,None
    elif method == "pca":
        pca=PCA(n_components=0.90)
        x_train_pca=pca.fit_transform(x_train_norm)
        x_val_pca=pca.transform(x_val_norm)
        x_test_pca=pca.transform(x_test_norm)
        return x_train_pca, x_val_pca, x_test_pca,scaler,scaler,pca
    elif method == "relief":
        weights,ranking=calculo.reliefF(x_train_norm,y_train,n_neighbors=10)
        top_15_idx=ranking[:15]
        x_train_relief=x_train_norm[:,top_15_idx]
        x_val_relief=x_val_norm[:,top_15_idx]
        x_test_relief=x_test_norm[:,top_15_idx]
        return x_train_relief, x_val_relief, x_test_relief,scaler,top_15_idx
    else:
        raise ValueError("Método desconhecido. Use 'all', 'pca' ou 'relief'.")
