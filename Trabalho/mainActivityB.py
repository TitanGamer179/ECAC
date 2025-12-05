#Meta 2 do Projeto

import os
import openfile
import graficoB
import calculoB
import calculo
import numpy as np

def main():
    datasets={}
    # =========================================================================
    # 0. LOAD DATA
    # =========================================================================
    dir_=os.path.dirname(os.path.abspath(__file__))
    dir_path=os.path.join(dir_,'diretoria')
    all_data = openfile.open_all_files(dir_path)
    if all_data.size == 0:
        print("Nenhum dado foi carregado. Verifique o caminho da diretoria.")
        return
    print("Ficheiros abertos com sucesso!")
    #all_data=calculo.add_magnitude(all_data)
    all_data=all_data[all_data[:, 11] <= 7] 
    print("Dados filtrados com sucesso!")
    feats_manual,lables_manual,parts_manual=calculo.extrair_features(all_data)
    segs_raw, labels_seg, parts_seg = calculo.segmentation(all_data)
    # =========================================================================
    # 1. DATA AUGMENTATION
    # =========================================================================
    print("INICIANDO DATA AUGMENTATION...\n")
    #1.1 Analise de Balanceamento
    print("Analisando balanceamento dos dados...")
    calculoB.check_balance(lables_manual)
    graficoB.plot_balance(lables_manual, title="Distribuição Original")
    #1.2 Create SMOTE
    print("Aplicando SMOTE para balanceamento dos dados...")
    _, lables_aug = calculoB.apply_smote(feats_manual, lables_manual)
    calculoB.check_balance(lables_aug)
    graficoB.plot_balance(lables_aug, title="Distribuição Após SMOTE")
    #1.3 Visualização de Augmentação
    print("Visualizando dados após augmentação...")
    mask_p3_a4 = (parts_manual == 3) & (lables_manual == 4)
    if np.sum(mask_p3_a4) > 0:
        feats_p3_a4 = feats_manual[mask_p3_a4]
        lables_p3_a4 = lables_manual[mask_p3_a4]
        synthetic_samples = calculoB.custom_smote(feats_p3_a4, lables_p3_a4, 4, N=3, k=2)
        if len(synthetic_samples) > 0:
            graficoB.plot_augmentation_scatter(feats_p3_a4, synthetic_samples)
        else:
            print("Não foi possível gerar amostras sintéticas (dados insuficientes).")
    else:
        print("Dados do Participante 3 / Atividade 4 não encontrados.")

    # =========================================================================
    # 2.Embedding features
    # =========================================================================
    print("Extraindo features de embedding...")
    #2.1 Extract embedding(30HZ)
    dataset_A=feats_manual
    dataset_B=calculoB.extract_embedding_features(segs_raw)
    #==========================================================================
    #3. Data splitting strategy
    #==========================================================================
    print("Dividindo os dados em treino, validação e teste...")
    #3.1 TVT Split
    print("Dividindo os dados usando a estratégia TVT Split...")
    #Dataset B
    train_idx_B, val_idx_B, test_idx_B = calculoB.tvt_split(dataset_B, labels_seg)
    X_train_seg = dataset_B[train_idx_B]
    y_train_seg = labels_seg[train_idx_B]
    X_val_seg   = dataset_B[val_idx_B]
    y_val_seg   = labels_seg[val_idx_B]
    X_test_seg  = dataset_B[test_idx_B]
    y_test_seg  = labels_seg[test_idx_B]
    #Dataset A
    train_idx_A, val_idx_A, test_idx_A = calculoB.tvt_split(dataset_A, lables_manual)
    X_train_manual = dataset_A[train_idx_A]
    y_train_manual = lables_manual[train_idx_A]
    X_val_manual   = dataset_A[val_idx_A]
    y_val_manual   = lables_manual[val_idx_A]
    X_test_manual  = dataset_A[test_idx_A]
    y_test_manual  = lables_manual[test_idx_A]
    print("Divisão concluída com sucesso!")
    #3.2 Subject Level Split
    print("Dividindo os dados usando a estratégia TVT Split por participante...")
    calculoB.split_between_subjects(parts_seg)
    #3.4 Method: "all";"pca";"relief"
    print("a)Usando método de seleção de features: All features")
    X_train_all_B, X_val_all_B, X_test_all_B, _, _ = calculoB.combined_split(X_train_seg, y_train_seg, X_val_seg, X_test_seg, method="all")
    X_train_all_A, X_val_all_A, X_test_all_A, _, _ = calculoB.combined_split(X_train_manual, y_train_manual, X_val_manual, X_test_manual, method="all")
    print("b)Usando método de seleção de features: pca features")
    X_train_pca_B, X_val_pca_B, X_test_pca_B, _, _ = calculoB.combined_split(X_train_seg, y_train_seg, X_val_seg, X_test_seg, method="pca")
    X_train_pca_A, X_val_pca_A, X_test_pca_A, _, _ = calculoB.combined_split(X_train_manual, y_train_manual, X_val_manual, X_test_manual, method="pca")
    print("c)Usando método de seleção de features: relief features")
    X_train_rel_B, X_val_rel_B, X_test_rel_B, _, _ = calculoB.combined_split(X_train_seg, y_train_seg, X_val_seg, X_test_seg, method="relief")
    X_train_rel_A, X_val_rel_A, X_test_rel_A, _, _ = calculoB.combined_split(X_train_manual, y_train_manual, X_val_manual, X_test_manual, method="relief")
    print("Seleção de features concluída com sucesso!")
    #==========================================================================
    #4. Model learning
    #==========================================================================
    print("\n=== 4. MODEL LEARNING (Exemplo Dataset B) ===")
    k_value = 3
    print(f"Treinando k-NN (k={k_value}) - All Features...")
    _, preds_all = calculoB.train_evaluate_knn(X_train_all_B, y_train_seg, X_val_all_B, y_val_seg, X_test_all_B, y_test_seg, k_value)
    print(f"Treinando k-NN (k={k_value}) - PCA...")
    _, preds_pca = calculoB.train_evaluate_knn(X_train_pca_B, y_train_seg, X_val_pca_B, y_val_seg, X_test_pca_B, y_test_seg, k_value)
    print(f"Treinando k-NN (k={k_value}) - ReliefF...")
    _, preds_rel = calculoB.train_evaluate_knn(X_train_rel_B, y_train_seg, X_val_rel_B, y_val_seg, X_test_rel_B, y_test_seg, k_value)
    # 4.2 Métricas
    print("\n=== MÉTRICAS (Dataset B) ===")
    print("--- All Features ---")
    calculoB.calculate_metrics(y_test_seg, preds_all)
    print("\n--- PCA ---")
    calculoB.calculate_metrics(y_test_seg, preds_pca)
    print("\n--- ReliefF ---")
    calculoB.calculate_metrics(y_test_seg, preds_rel)
    dataset={
        "Dataset_A": {
            "X_train": X_train_manual,
            "y_train": y_train_manual,
            "X_val": X_val_manual,
            "y_val": y_val_manual,
            "X_test": X_test_manual,
            "y_test": y_test_manual
        },
        "Dataset_B": {
            "X_train": X_train_seg,
            "y_train": y_train_seg,
            "X_val": X_val_seg,
            "y_val": y_val_seg,
            "X_test": X_test_seg,
            "y_test": y_test_seg
        }
    }
    #=========================================================================
    #5. Evaluation
    #=========================================================================
    #=========================================================================
    #6. Deployment
    #=========================================================================
    #=========================================================================
    #7. Go further
    #=========================================================================
    
    

if __name__ == "__main__":
    main()