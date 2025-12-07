import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from scipy.signal import resample
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy import stats
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
def extract_embedding_features(features):
    print("\n" + "="*80)
    print("EXTRAINDO EMBEDDINGS DO HARNET5...")
    print("="*80)
    print(f"[DEBUG] Número de segmentos: {len(features)}")
    if len(features) > 0:
        print(f"[DEBUG] Forma de cada segmento: {features[0].shape}")
    
    feature_encoder = embeddings_extractor.load_model()
    processed_features = []
    fs_original = 51.5
    
    for i, segment in enumerate(features):
        acc_xyz = segment[:, 1:4]
        acc_resampled, _ = embeddings_extractor.resample_to_30hz_5s(acc_xyz, fs_original)
        processed_features.append(acc_resampled)
        
        if i == 0:
            print(f"[DEBUG] Primeiro segmento resampled shape: {acc_resampled.shape}")
    
    x_all = np.transpose(np.array(processed_features), (0, 2, 1))
    print(f"[DEBUG] x_all shape antes do modelo: {x_all.shape}")
    print(f"[DEBUG] dtype: {x_all.dtype}")
    
    embeddings_list = []
    batch_size = 32
    with torch.no_grad():
        for batch_idx in range(0, x_all.shape[0], batch_size):
            batch = x_all[batch_idx:batch_idx+batch_size]
            batch_tensor = torch.tensor(batch, dtype=torch.float32)
            embeddings = feature_encoder(batch_tensor)
            embeddings_list.append(embeddings.cpu().numpy())
            
            if batch_idx == 0:
                print(f"[DEBUG] Embedding shape (primeiro batch): {embeddings.shape}")
    
    final_embeddings = np.concatenate(embeddings_list, axis=0)
    print(f"[DEBUG] Final embeddings shape antes reshape: {final_embeddings.shape}")
    print(f"[DEBUG] Mín: {final_embeddings.min():.4f}, Máx: {final_embeddings.max():.4f}, Média: {final_embeddings.mean():.4f}")
    
    if final_embeddings.ndim > 2:
        final_embeddings = final_embeddings.reshape(final_embeddings.shape[0], -1)
    
    print(f"[DEBUG] Final embeddings shape FINAL: {final_embeddings.shape}")
    print(f"✓ Extração de embeddings concluída com sucesso!")
    print("="*80 + "\n")
    return final_embeddings

# Requisito 3.1: TVT Split 60-20-20 Estratificado
def tvt_split(features, labels, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_state=None):
    """Split estratificado que mantém as proporções das classes em cada subset."""
    test=StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_state)
    train_val_idx, test_idx = next(test.split(features, labels))
    x_train_val=features[train_val_idx]
    y_train_val=labels[train_val_idx]
    original_idx_train_val=train_val_idx
    val=StratifiedShuffleSplit(n_splits=1, test_size=val_ratio/(train_ratio + val_ratio), random_state=random_state)
    train_idx, val_idx = next(val.split(x_train_val, y_train_val))
    final_train_idx=original_idx_train_val[train_idx]
    final_val_idx=original_idx_train_val[val_idx]
    n_total=len(features)
    print(f"Split Within-Subject realizado:")
    print(f"  Treino: {len(final_train_idx)} ({len(final_train_idx)/n_total:.1%}) -> Esperado 60%")
    print(f"  Valid:  {len(final_val_idx)} ({len(final_val_idx)/n_total:.1%}) -> Esperado 20%")
    print(f"  Teste:  {len(test_idx)} ({len(test_idx)/n_total:.1%}) -> Esperado 20%")
    return final_train_idx, final_val_idx, test_idx

# Requisito 3.2: TVT Split por Participante (Dinâmico para qualquer número)
def split_between_subjects(parts, random_state=None):
    """Split entre participantes com randomização a cada iteração.
    
    Args:
        parts: Array com IDs dos participantes
        random_state: None para randomização a cada iteração, ou int para seed fixo
    """
    unique_parts = np.unique(parts)
    if random_state is not None:
        np.random.seed(random_state)
    shuffled_parts = np.random.permutation(unique_parts)
    n_parts = len(unique_parts)
    train_size = int(0.6 * n_parts)
    val_size = int(0.2 * n_parts)
    train_p_ids = shuffled_parts[:train_size]
    val_p_ids = shuffled_parts[train_size:train_size+val_size]
    test_p_ids = shuffled_parts[train_size+val_size:]
    print(f"Participantes atribuídos (60-20-20 dinâmico):")
    print(f"  Treino: {sorted([int(x) for x in train_p_ids])} ({len(train_p_ids)}/{n_parts})")
    print(f"  Validação: {sorted([int(x) for x in val_p_ids])} ({len(val_p_ids)}/{n_parts})")
    print(f"  Teste: {sorted([int(x) for x in test_p_ids])} ({len(test_p_ids)}/{n_parts})")
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
    
# Requisito 5.1: Hyperparameter tuning

def hyperparameter_tuning(X_train, y_train, X_val, y_val, X_test, y_test, k_range=None):
    
    #encontrarmos o melhor k com train and validation data
    if k_range is None:
        k_range = list(range(1, 21))
        
    best_val_k= None
    best_val_acc=0
    val_accuracy=[]
    for k in k_range:
        knn= KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_val_preds= knn.predict(X_val)
        val_acc= accuracy_score(y_val, y_val_preds)
        val_accuracy.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc= val_acc
            best_val_k= k
            
        if k%5==1 or k==best_val_k:
            print(f"k={k}: Accuracy na Validação = {val_acc:.4f}")
        
    print(f"Melhor k na Validação: {best_val_k} com Accuracy = {best_val_acc:.4f}")
    
    #retreinar o dataset com este k
    
    X_train_val= np.vstack((X_train, X_val))
    y_train_val= np.hstack((y_train, y_val))
    
    best_model= KNeighborsClassifier(n_neighbors=best_val_k)
    best_model.fit(X_train_val, y_train_val)
    print("Modelo retreinado com os dados de Treino + Validação.")
    
    #avaliação final com os dataset treinados
    
    y_test_preds= best_model.predict(X_test)
    test_metrics= {
        "accuracy": accuracy_score(y_test, y_test_preds), 
        "precision": precision_score(y_test, y_test_preds, average='weighted', zero_division=0),
        "recall": recall_score(y_test, y_test_preds, average='weighted', zero_division=0),
        "f1_score": f1_score(y_test, y_test_preds, average='weighted',),
        "confusion_matrix": confusion_matrix(y_test, y_test_preds)}
    
    print(f"Accuracy no Teste com k={best_val_k}: {test_metrics['accuracy']:.4f}")
    print(f"F1-Score no Teste com k={best_val_k}: {test_metrics['f1_score']:.4f}")
    return best_val_k, test_metrics, val_accuracy, best_model,y_test_preds

# SVM com kernel rbf 
def hyperparameter_tuning_svm(X_train, y_train, X_val, y_val, X_test, y_test):
    print("A testar svm com o kernel rbf:")
    print("="*80)
    
    # Grid de hiperparâmetros para SVM
    C_values = [0.1, 1, 10, 100]
    gamma_values = ['scale', 'auto', 0.001, 0.01, 0.1]
    
    best_val_acc = 0
    best_params = {'C': None, 'gamma': None}
    best_model_svm = None
    
    print(f"\n[SVM] A testar {len(C_values) * len(gamma_values)} combinações")
    
    for C in C_values:
        for gamma in gamma_values:
            svm = SVC(kernel='rbf', C=C, gamma=gamma, probability=True, random_state=42)
            svm.fit(X_train, y_train)
            y_val_preds = svm.predict(X_val)
            val_acc = accuracy_score(y_val, y_val_preds)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = {'C': C, 'gamma': gamma}
                best_model_svm = svm
    
    print(f"[SVM] Melhores parâmetros: C={best_params['C']}, gamma={best_params['gamma']}")
    print(f"[SVM] Accuracy na validação: {best_val_acc:.4f}")
    
    # Retreinar com dados de treino + validação
    X_train_val = np.vstack((X_train, X_val))
    y_train_val = np.hstack((y_train, y_val))
    
    best_model_svm = SVC(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'], 
                         probability=True, random_state=42)
    best_model_svm.fit(X_train_val, y_train_val)
    
    # Avaliação final
    y_test_preds = best_model_svm.predict(X_test)
    test_metrics = {
        "accuracy": accuracy_score(y_test, y_test_preds),
        "precision": precision_score(y_test, y_test_preds, average='weighted', zero_division=0),
        "recall": recall_score(y_test, y_test_preds, average='weighted', zero_division=0),
        "f1_score": f1_score(y_test, y_test_preds, average='weighted', zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_test_preds)
    }
    
    print(f"[SVM] Accuracy no Teste: {test_metrics['accuracy']:.4f}")
    print(f"[SVM] F1-Score no Teste: {test_metrics['f1_score']:.4f}")
    
    return best_params, test_metrics, best_model_svm, y_test_preds

#melhoria da random forest
def hyperparameter_tuning_rf(X_train, y_train, X_val, y_val, X_test, y_test):
    print("A testar Random Forest:")
    
    # Grid de hiperparâmetros para Random Forest
    n_estimators_values = [50, 100, 200]
    max_depth_values = [10, 20, None]  # não definimos nenhum limite
    best_val_acc = 0
    best_params = {'n_estimators': None, 'max_depth': None}
    best_model_rf = None
    
    print(f"\n[RF] A testar {len(n_estimators_values) * len(max_depth_values)} combinações")
    
    for n_est in n_estimators_values:
        for depth in max_depth_values:
            rf = RandomForestClassifier(n_estimators=n_est, max_depth=depth, 
                                       random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            y_val_preds = rf.predict(X_val)
            val_acc = accuracy_score(y_val, y_val_preds)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = {'n_estimators': n_est, 'max_depth': depth}
                best_model_rf = rf
    
    print(f"[RF] Melhores parâmetros: n_estimators={best_params['n_estimators']}, "
          f"max_depth={best_params['max_depth']}")
    print(f"[RF] Accuracycia na validação: {best_val_acc:.4f}")
    
    # Retreinar com dados de treino + validação
    X_train_val = np.vstack((X_train, X_val))
    y_train_val = np.hstack((y_train, y_val))
    
    best_model_rf = RandomForestClassifier(n_estimators=best_params['n_estimators'],
                                           max_depth=best_params['max_depth'],
                                           random_state=42, n_jobs=-1)
    best_model_rf.fit(X_train_val, y_train_val)
    
    # Avaliação final
    y_test_preds = best_model_rf.predict(X_test)
    test_metrics = {
        "accuracy": accuracy_score(y_test, y_test_preds),
        "precision": precision_score(y_test, y_test_preds, average='weighted', zero_division=0),
        "recall": recall_score(y_test, y_test_preds, average='weighted', zero_division=0),
        "f1_score": f1_score(y_test, y_test_preds, average='weighted', zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_test_preds),
        "feature_importance": best_model_rf.feature_importances_
    }
    
    print(f"[RF] Accuracy no Teste: {test_metrics['accuracy']:.4f}")
    print(f"[RF] F1-Score no Teste: {test_metrics['f1_score']:.4f}")
    
    # Mostrar top 10 features mais importantes
    if hasattr(best_model_rf, 'feature_importances_'):
        top_features = np.argsort(best_model_rf.feature_importances_)[-10:][::-1]
        print(f"[RF] Top 10 features mais importantes:")
        for idx, feat_idx in enumerate(top_features, 1):
            print(f"     {idx}. Feature {feat_idx}: {best_model_rf.feature_importances_[feat_idx]:.4f}")
    
    return best_params, test_metrics, best_model_rf, y_test_preds

# LOSO
def loso_cross_validation(X_all, y_all, participants_all, model_type='knn', k=5):
    print("A testar LOSO")
    print(f"Modelo: {model_type.upper()}")
    print(f"Testar com {len(np.unique(participants_all))} participantes")
    
    unique_subjects = np.unique(participants_all)
    accuracies_loso = []
    f1_scores_loso = []
    
    for test_subject in unique_subjects:
        # Dividir: treino (todos exceto subject), teste (apenas subject)
        test_mask = participants_all == test_subject
        train_mask = ~test_mask
        
        X_train_loso = X_all[train_mask]
        y_train_loso = y_all[train_mask]
        X_test_loso = X_all[test_mask]
        y_test_loso = y_all[test_mask]
        
        # Normalizar
        scaler_loso = StandardScaler()
        X_train_loso = scaler_loso.fit_transform(X_train_loso)
        X_test_loso = scaler_loso.transform(X_test_loso)
        
        # Treinar modelo apropriado
        if model_type == 'knn':
            model = KNeighborsClassifier(n_neighbors=k)
        elif model_type == 'svm':
            model = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
        elif model_type == 'rf':
            model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
        else:
            raise ValueError(f"Model type '{model_type}' não reconhecido")
        
        model.fit(X_train_loso, y_train_loso)
        
        # Avaliar
        y_pred_loso = model.predict(X_test_loso)
        acc_loso = accuracy_score(y_test_loso, y_pred_loso)
        f1_loso = f1_score(y_test_loso, y_pred_loso, average='weighted', zero_division=0)
        
        accuracies_loso.append(acc_loso)
        f1_scores_loso.append(f1_loso)
    
    # Estatísticas finais
    mean_acc = np.mean(accuracies_loso)
    std_acc = np.std(accuracies_loso)
    mean_f1 = np.mean(f1_scores_loso)
    std_f1 = np.std(f1_scores_loso)
    
    print(f"\n[LOSO] Resultados por participante:")
    for idx, subject in enumerate(unique_subjects):
        print(f"       Participante {int(subject):2d}: Acc = {accuracies_loso[idx]:.4f}, "
              f"F1 = {f1_scores_loso[idx]:.4f}")
    
    print(f"\n[LOSO] Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"[LOSO] F1-Score: {mean_f1:.4f} ± {std_f1:.4f}")
    print("="*80 + "\n")
    
    loso_metrics = {
        'accuracies': accuracies_loso,
        'f1_scores': f1_scores_loso,
        'mean_acc': mean_acc,
        'std_acc': std_acc,
        'mean_f1': mean_f1,
        'std_f1': std_f1
    }
    
    return loso_metrics

# Platt scalling
def train_evaluate_with_calibration(X_train, y_train, X_val, y_val, X_test, y_test, model_type='knn', k=5):
    print("A implementar Platt Scalling:")
    print(f"Modelo Base: {model_type.upper()}")
    
    # Criar modelo base
    if model_type == 'knn':
        base_model = KNeighborsClassifier(n_neighbors=k)
    elif model_type == 'svm':
        base_model = SVC(kernel='rbf', C=10, gamma='scale')
    elif model_type == 'rf':
        base_model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
    else:
        raise ValueError(f"Model type '{model_type}' não reconhecido")
    
    # Calibrar usando dados de validação
    calibrated_model = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
    
    X_train_val = np.vstack((X_train, X_val))
    y_train_val = np.hstack((y_train, y_val))
    
    calibrated_model.fit(X_train_val, y_train_val)
    
    # Predições e confiança
    y_pred = calibrated_model.predict(X_test)
    y_pred_proba = calibrated_model.predict_proba(X_test)
    
    # Calcular confiança (probabilidade máxima)
    confidence = np.max(y_pred_proba, axis=1)
    mean_confidence = np.mean(confidence)
    
    # Métricas
    test_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average='weighted', zero_division=0),
        "mean_confidence": mean_confidence,
        "min_confidence": np.min(confidence),
        "max_confidence": np.max(confidence)
    }
    
    print(f"\n[CALIB] Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"[CALIB] F1-Score: {test_metrics['f1_score']:.4f}")
    print(f"[CALIB] Confiança Média: {test_metrics['mean_confidence']:.4f}")
    print(f"[CALIB] Confiança Min-Max: [{test_metrics['min_confidence']:.4f}, {test_metrics['max_confidence']:.4f}]")
    print("="*80 + "\n")
    
    return calibrated_model, test_metrics, y_pred, confidence

        
# Requisito 5.2: Report and analysis of results


def report_results(datasets_dict, participants=None, n_iterations=1):
    results= {}
    predict={}

    dataset_names =['Dataset_A', 'Dataset_B']
    methods= ['all', 'pca', 'relief']
    splitting = ['Within-Subject', 'Between-Subject']
    
    for dataset_name in dataset_names: 
        dataset = datasets_dict[dataset_name]
        x_all=np.vstack((dataset['X_train'], dataset['X_val'], dataset['X_test']))
        y_all=np.hstack((dataset['y_train'], dataset['y_val'], dataset['y_test']))
        
        for method in methods:
            for split_strategy in splitting:
                config_name= f"{dataset_name}_{method}_{split_strategy}"
                print(f"\nA executar a configuração: {config_name}")
                
                accuracies= []
                all_metrics= []
                
                for iter in range(n_iterations):
                    if split_strategy == 'Within-Subject':
                        train_idx, val_idx, test_idx = tvt_split(x_all, y_all, random_state=None)
                    else:
                        if participants is None:
                            print("Erro: participantes não fornecidos para split Between-Subject.")
                            continue
                        train_idx, val_idx, test_idx = split_between_subjects(participants, random_state=None)
                    
                    
                    X_train= x_all[train_idx]
                    y_train= y_all[train_idx]
                    X_val= x_all[val_idx]
                    y_val= y_all[val_idx]
                    X_test= x_all[test_idx]
                    y_test= y_all[test_idx]
                    
                    X_train_fs, X_val_fs, X_test_fs, scaler, selector = combined_split(X_train, y_train, X_val, X_test, method)
                    
                    best_k, test_metrics, val_accuracy,best_model,y_test_preds = hyperparameter_tuning(X_train_fs, y_train, X_val_fs, y_val, X_test_fs, y_test)
                    
                    accuracies.append(test_metrics['accuracy'])
                    all_metrics.append(test_metrics)
                    
                    if iter ==0:
                        predict[config_name]= {'y_true': y_test, 'y_pred': y_test_preds, 'best_k': best_k}
                
                if accuracies: 
                    results[config_name]={ 'accuracies': accuracies,'mean_acc': np.mean(accuracies), 'std_acc': np.std(accuracies), 'metrics': all_metrics}
                    print(f"Configuração {config_name} - Acurácia Média: {np.mean(accuracies):.4f}, Desvio Padrão: {np.std(accuracies):.4f}")
                else:
                    print(f"Nenhum resultado obtido para a configuração {config_name}.")
                                    
    sorted_results= sorted(results.items(), key=lambda item: item[1]['mean_acc'], reverse=True)
    print("\n=== Resultados Finais Ordenados por Acurácia Média ===")
    for rank, (name,metrics) in enumerate(sorted_results,1):
        print(f"{rank}. {name} - Accuracy Média: {metrics['mean_acc']:.4f}, Desvio Padrão: {metrics['std_acc']:.4f}")
    return results, predict

# Requisito 5.3: Hypothesis Testing
def analyze_results_5_3(results, predictions):
    results_by_dataset = {}
    results_by_method = {}
    results_by_strategy = {}
    
    for config_name, metrics in results.items():
        parts = config_name.split('_')
        dataset = parts[0] + '_' + parts[1]  # Dataset_A ou Dataset_B
        method = parts[2]  # all, pca, relief
        strategy = '_'.join(parts[3:])  # Within-Subject ou Between-Subject
        
        mean_acc = metrics['mean_acc']
        std_acc = metrics['std_acc']
        
        # Agrupar por dataset
        if dataset not in results_by_dataset:
            results_by_dataset[dataset] = []
        results_by_dataset[dataset].append((config_name, mean_acc, std_acc))
        
        # Agrupar por método
        if method not in results_by_method:
            results_by_method[method] = []
        results_by_method[method].append((config_name, mean_acc, std_acc))
        
        # Agrupar por estratégia
        if strategy not in results_by_strategy:
            results_by_strategy[strategy] = []
        results_by_strategy[strategy].append((config_name, mean_acc, std_acc))
    
    print("\nComparação por aataset:")
    for dataset, confs in results_by_dataset.items():
        accs = [acc for _, acc, _ in confs]
        print(f"  {dataset}: Média = {np.mean(accs):.4f}, Min = {np.min(accs):.4f}, Max = {np.max(accs):.4f}")
    
    print("\nComparação por método de seleção de features:")
    for method, confs in results_by_method.items():
        accs = [acc for _, acc, _ in confs]
        print(f"  {method:10s}: Média = {np.mean(accs):.4f}, Min = {np.min(accs):.4f}, Max = {np.max(accs):.4f}")
    
    print("\nComparação por estratégia de split")
    for strategy, confs in results_by_strategy.items():
        accs = [acc for _, acc, _ in confs]
        print(f"  {strategy:20s}: Média = {np.mean(accs):.4f}, Min = {np.min(accs):.4f}, Max = {np.max(accs):.4f}")
    
    print("\n Melhor e Pior Configuração")
    sorted_configs = sorted(results.items(), key=lambda x: x[1]['mean_acc'], reverse=True)

    print("\n5. Análise do desvio padrão:")
    std_values = [m['std_acc'] for m in results.values()]
    print(f"  Desvio padrão médio: {np.mean(std_values):.4f}")
    print(f"  Configuração mais estável: {min(results.items(), key=lambda x: x[1]['std_acc'])[0]}")
    print(f"  Configuração menos estável: {max(results.items(), key=lambda x: x[1]['std_acc'])[0]}")
    

    print("\n6. Teste de Friedman")  
    # Preparar dados para o teste de Friedman
    config_names_list = list(results.keys())
    accuracies_data = []
    
    for config_name in config_names_list:
        if 'accuracies' in results[config_name]:
            accs = results[config_name]['accuracies']
        else:
            accs = [results[config_name]['mean_acc']]
        accuracies_data.append(accs)
    
    # Garantir mesmo tamanho de amostras (necessário para Friedman)
    min_samples = min(len(accs) for accs in accuracies_data)
    accuracies_data = [accs[:min_samples] for accs in accuracies_data]
    
    # Transpor para formato correto (linhas = iterações, colunas = configurações)
    accuracies_matrix = np.array(accuracies_data).T
    
    # Executar teste de Friedman
    friedman_stat, friedman_p = stats.friedmanchisquare(*accuracies_data)
    
    print(f"\n  Estatística de Friedman: {friedman_stat:.4f}")
    print(f"  p-value: {friedman_p:.6f}")
    
    if friedman_p < 0.05:
        print(f"Há diferenças significativas (p < 0.05)")
        print(f"Rejeitar H0: Há diferenças significativas entre configurações")
        print(f"Vamos utilizar o teste de pares para obter mais detalhes")
    else:
        print(f"Não há diferenças significativas (p ≥ 0.05)")
        print(f"Não rejeitar H0, as configurações não diferem significativamente")
    
    # Calcular rankings de Friedman (média dos ranks)
    print("\n  Rankings de Friedman:")
    
    # Ranquear cada iteração
    ranks_all = []
    for row in accuracies_matrix:
        ranks = stats.rankdata(row)
        ranks_all.append(ranks)
    
    # Média dos rankings
    mean_ranks = np.mean(ranks_all, axis=0)
    
    # Criar lista com nomes e ranks
    ranked_configs = list(zip(config_names_list, mean_ranks))
    ranked_configs.sort(key=lambda x: x[1])  # Ordenar por rank (menor é melhor)
    
    for i, (config_name, mean_rank) in enumerate(ranked_configs, 1):
        mean_acc = results[config_name]['mean_acc']
        print(f"    {i}. {config_name:40s} | Rank: {mean_rank:5.2f} | Accuracya: {mean_acc:.4f}")
    
    # Requisito 6: Deployement

def deploy_model(raw_data, model, scaler, feature_extractor_func= None, selector= None, selector_type= None):
    if feature_extractor_func:
        features = feature_extractor_func(raw_data)
    else:
        acc_data= raw_data[:, 0:3]
        acc_resample= embeddings_extractor.resample_to_30hz_5s(acc_data, fs_original=51.5)
        feature_encoder= embeddings_extractor.load_model()
        
        x_input= np.transpose(acc_resample[np.newaxis, :, :], (0, 2, 1))
        x_tensor= torch.tensor(x_input, dtype= torch.float32)
        
        with torch.no_grad():
            embeddings= feature_encoder(x_tensor).cpu().numpy()
        features= embeddings.reshape(1, -1)[0]
        
    features_norm= scaler.transform(features.reshape(1,-1))
    
    if selector is not None:
        if selector_type == 'pca':
            features_final= selector.transform(features_norm)
        elif selector_type == 'relief':
            features_final= features_norm[:, selector]
    else:
        features_final= features_norm
    
    prediction= model.predict_proba(features_final)[0]
    proba= model.predict_proba(features_final)[0]
    confidence= np.max(proba)
    
    return int(prediction), confidence

def deploy_pipeline(best_model, datasets):
    config = best_model['config']
    model= best_model['model']
    scaler= best_model['scaler']
    selector= best_model['selector']
    selector_type= best_model['selector_type']
    
    def predict_activity(raw_data_shape):
        return deploy_model(raw_data_shape, model, scaler, feature_extractor_func= config.get('feature_extractor'), selector= selector, selector_type= selector_type)
    
    print("\nPipeline de deploy criada com sucesso!")
    return predict_activity, config


    # 7. Conclusões
    print("\n7. Conclusão:")
    best_config, best_metrics = sorted_configs[0]
    worst_config, worst_metrics = sorted_configs[-1]
    
    print(f"   Melhor configuração: {best_config}")
    print(f"    Accuracy: {best_metrics['mean_acc']:.4f} (±{best_metrics['std_acc']:.4f})")
    print(f"    Precision: {best_metrics['metrics'][0]['precision']:.4f}")
    print(f"    Recall: {best_metrics['metrics'][0]['recall']:.4f}")
    print(f"    F1-Score: {best_metrics['metrics'][0]['f1_score']:.4f}")
    
    print(f"\n  Pior configuração: {worst_config}")
    print(f"    Accuracy: {worst_metrics['mean_acc']:.4f} (±{worst_metrics['std_acc']:.4f})")
    
    gap = best_metrics['mean_acc'] - worst_metrics['mean_acc']
    print(f"\n Diferença entre melhor e pior: {gap:.4f} ({gap*100:.2f}%)")
    
    print(f"\n Teste de Friedman:")
    if friedman_p < 0.05:
        print(f"    • Diferenças globais significativas (p = {friedman_p:.6f})")
        print(f"    • Há diferenças significativas entre as configurações")
    else:
        print(f"    • Diferenças globais não significativas (p = {friedman_p:.6f})")
        print(f"    • Todas as configurações têm desempenho estatisticamente similar")
 
    print("\n7.1. A utilizar predictions")
    if predictions:
        for config_name in sorted(predictions.keys()):
            pred_data = predictions[config_name]
            y_true = pred_data['y_true']
            y_pred = pred_data['y_pred']
            
            # Calcular métricas das predições
            acc = accuracy_score(y_true, y_pred)
            conf_mat = confusion_matrix(y_true, y_pred)
            
            print(f"\n  {config_name}:")
            print(f"    Accuracy: {acc:.4f}")
            print(f"    Matriz de confusão shape: {conf_mat.shape}")
            print(f"    Verdadeiros Positivos: {np.trace(conf_mat)}")
    else:
        print("Nenhuma prediction disponível para análise")
    

    print("\n8. Teste T de Student Pareado")
    print("\nHipótese Nula (H0): Não há diferença significativa de accuracy entre pares")
    print("Hipótese Alternativa (H1): Há diferença significativa")
    print("Nível de Significância: α = 0.05")
    
    # Extrair acurácias de cada configuração
    config_names_list = list(results.keys())
    accuracies_dict = {}
    
    for config_name in config_names_list:
        # Se há múltiplas iterações, usar a lista de accuracies
        if 'accuracies' in results[config_name]:
            accuracies_dict[config_name] = results[config_name]['accuracies']
        else:
            # Fallback: usar apenas a acurácia média
            accuracies_dict[config_name] = [results[config_name]['mean_acc']]
    
    # Comparações principais
    print("\n8.1. Comparação entre datasets")
    
    # Dataset_A vs Dataset_B, comparar o mesmo método e estratégia
    dataset_pairs = []
    for method in ['all', 'pca', 'relief']:
        for strategy in ['Within-Subject', 'Between-Subject']:
            config_a = f"Dataset_A_{method}_{strategy}"
            config_b = f"Dataset_B_{method}_{strategy}"
            
            if config_a in accuracies_dict and config_b in accuracies_dict:
                accs_a = np.array(accuracies_dict[config_a])
                accs_b = np.array(accuracies_dict[config_b])
                
                # Garantir mesmo tamanho para teste pareado
                min_len = min(len(accs_a), len(accs_b))
                accs_a = accs_a[:min_len]
                accs_b = accs_b[:min_len]
                
                # Teste T de Student pareado
                t_stat, p_value = stats.ttest_rel(accs_a, accs_b)
                mean_diff = np.mean(accs_a) - np.mean(accs_b)
                
                dataset_pairs.append({
                    'comparison': f"{config_a} vs {config_b}",
                    't_stat': t_stat,
                    'p_value': p_value,
                    'mean_diff': mean_diff,
                    'significant': p_value < 0.05
                })
                
                sig_marker = "significativa" if p_value < 0.05 else "✗ no_significativa"
                better_config = config_a if mean_diff > 0 else config_b
                print(f"\n  {config_a} vs {config_b}")
                print(f"    t-statistic: {t_stat:8.4f}")
                print(f"    p-value:     {p_value:8.6f} {sig_marker}")
                print(f"    Diferença:   {mean_diff:8.4f}")
                if p_value < 0.05:
                    print(f"{better_config} é significativamente melhor")
    
    print("\n8.2.Comparação entre os métodos")
    
    # Para cada dataset e estratégia, comparar métodos
    method_pairs = []
    for dataset in ['Dataset_A', 'Dataset_B']:
        for strategy in ['Within-Subject', 'Between-Subject']:
            methods = ['all', 'pca', 'relief']
            
            for i in range(len(methods)):
                for j in range(i+1, len(methods)):
                    config_i = f"{dataset}_{methods[i]}_{strategy}"
                    config_j = f"{dataset}_{methods[j]}_{strategy}"
                    
                    if config_i in accuracies_dict and config_j in accuracies_dict:
                        accs_i = np.array(accuracies_dict[config_i])
                        accs_j = np.array(accuracies_dict[config_j])
                        
                        # Garantir mesmo tamanho
                        min_len = min(len(accs_i), len(accs_j))
                        accs_i = accs_i[:min_len]
                        accs_j = accs_j[:min_len]
                        
                        # Teste T de Student pareado
                        t_stat, p_value = stats.ttest_rel(accs_i, accs_j)
                        mean_diff = np.mean(accs_i) - np.mean(accs_j)
                        
                        method_pairs.append({
                            'comparison': f"{methods[i]} vs {methods[j]} ({dataset}, {strategy})",
                            't_stat': t_stat,
                            'p_value': p_value,
                            'mean_diff': mean_diff,
                            'significant': p_value < 0.05
                        })
                        
                        sig_marker = "significativa" if p_value < 0.05 else "no_significativa"
                        better_config = config_i if mean_diff > 0 else config_j
                        print(f"\n  {config_i} vs {config_j}")
                        print(f"    t-statistic: {t_stat:8.4f}")
                        print(f"    p-value:     {p_value:8.6f} {sig_marker}")
                        print(f"    Diferença:   {mean_diff:8.4f}")
                        if p_value < 0.05:
                            print(f"{better_config.split('_')[1]} é significativamente melhor")
    
    print("\n8.3.Comparação entre estratégiass") 
    # Para cada dataset e método, comparar estratégias
    strategy_pairs = []
    for dataset in ['Dataset_A', 'Dataset_B']:
        for method in ['all', 'pca', 'relief']:
            config_within = f"{dataset}_{method}_Within-Subject"
            config_between = f"{dataset}_{method}_Between-Subject"
            
            if config_within in accuracies_dict and config_between in accuracies_dict:
                accs_within = np.array(accuracies_dict[config_within])
                accs_between = np.array(accuracies_dict[config_between])
                
                # Garantir mesmo tamanho
                min_len = min(len(accs_within), len(accs_between))
                accs_within = accs_within[:min_len]
                accs_between = accs_between[:min_len]
                
                # Teste T de Student pareado
                t_stat, p_value = stats.ttest_rel(accs_within, accs_between)
                mean_diff = np.mean(accs_within) - np.mean(accs_between)
                
                strategy_pairs.append({
                    'comparison': f"Within vs Between ({dataset}, {method})",
                    't_stat': t_stat,
                    'p_value': p_value,
                    'mean_diff': mean_diff,
                    'significant': p_value < 0.05
                })
                
                sig_marker = "significativa" if p_value < 0.05 else "no_significativa"
                better_config = "Within-Subject" if mean_diff > 0 else "Between-Subject"
                print(f"\n  {config_within} vs {config_between}")
                print(f"    t-statistic: {t_stat:8.4f}")
                print(f"    p-value:     {p_value:8.6f} {sig_marker}")
                print(f"    Diferença:   {mean_diff:8.4f}")
                if p_value < 0.05:
                    print(f"{better_config} é significativamente melhor")
    
    print("\n8.4.Resumo das estatísticas")
    
    all_pairs = dataset_pairs + method_pairs + strategy_pairs
    sig_count = sum(1 for p in all_pairs if p['significant'])
    total_count = len(all_pairs)
    
    print(f"\n  Total de comparações pareadas: {total_count}")
    print(f"  Comparações significativas (p < 0.05): {sig_count}")
    print(f"  Comparações não significativas: {total_count - sig_count}")
    
    print("\n  Comparações significativas")
    print("  " + "-" * 76)
    sig_pairs = [p for p in all_pairs if p['significant']]
    if sig_pairs:
        for pair in sorted(sig_pairs, key=lambda x: x['p_value']):
            print(f"    {pair['comparison']:<50} p={pair['p_value']:.6f}, t={pair['t_stat']:7.4f}, diff={pair['mean_diff']:7.4f}")
    else:
        print("Nenhuma diferença significativa encontrada")
    
    
    




    
