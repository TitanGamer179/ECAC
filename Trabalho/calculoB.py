import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.signal import resample
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import random
import calculo
import torch
import embeddings_extractor

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
    print("Carregando modelo Harnet5...")
    feature_encoder = embeddings_extractor.load_model()
    processed_features = []
    fs_original = 51.5
    for segment in features:
        acc_xyz=segment[:,1:4]
        acc_resampled, _ = embeddings_extractor.resample_to_30hz_5s(acc_xyz, fs_original)
        processed_features.append(acc_resampled)
    x_all = np.transpose(np.array(processed_features), (0, 2, 1))
    embeddings_list=[]
    batch_size = 32
    with torch.no_grad():
        for i in range(0, x_all.shape[0], batch_size):
            batch = x_all[i:i+batch_size]
            batch_tensor = torch.tensor(batch, dtype=torch.float32)
            embeddings = feature_encoder(batch_tensor).numpy()
            embeddings_list.append(embeddings.cpu().numpy())
    final_embeddings=np.concatenate(embeddings_list, axis=0)
    return final_embeddings

# Requisito 3.1: TVT Split 60-20-20 por Participante
def tvt_split(features, labels, train_ratio=0.6, val_ratio=0.2):
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
def split_between_subjects(parts):
    unique_parts = np.unique(parts)
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
        return x_train_pca, x_val_pca, x_test_pca,scaler,pca
    elif method == "relief":
        _,ranking=calculo.relieff(x_train_norm,y_train,k=10)
        top_15_idx=ranking[:15]
        x_train_relief=x_train_norm[:,top_15_idx]
        x_val_relief=x_val_norm[:,top_15_idx]
        x_test_relief=x_test_norm[:,top_15_idx]
        return x_train_relief, x_val_relief, x_test_relief,scaler,top_15_idx
    else:
        raise ValueError("Método desconhecido. Use 'all', 'pca' ou 'relief'.")

# Requisito 4.1 : Treinamento e Avaliação do k-NN
def train_evaluate_knn(X_train, y_train, X_val, y_val, X_test, y_test, k):
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X_train)
    _, val_indices = knn.kneighbors(X_val)
    val_predictions = []
    for indices in val_indices:
        neighbor_labels = y_train[indices].astype(int)
        pred_label = np.bincount(neighbor_labels).argmax()
        val_predictions.append(pred_label)
    val_accuracy = np.mean(np.array(val_predictions) == y_val.astype(int))
    print(f"Acurácia na validação com k={k}: {val_accuracy:.4f}")
    
    _, test_indices = knn.kneighbors(X_test)
    test_predictions = []
    for indices in test_indices:
        neighbor_labels = y_train[indices].astype(int)
        pred_label = np.bincount(neighbor_labels).argmax()
        test_predictions.append(pred_label)
    test_accuracy = np.mean(np.array(test_predictions) == y_test.astype(int))
    print(f"Acurácia no teste com k={k}: {test_accuracy:.4f}")
    
    return knn, np.array(test_predictions)

# Requisito 4.2: Cálculo de Métricas de Avaliação
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred)
    print(f"Acurácia: {accuracy:.4f}")
    print(f"Precisão: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("Matriz de Confusão:")
    print(conf_matrix)