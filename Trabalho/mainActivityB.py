#Meta 2 do Projeto

import numpy as np
import os
import openfile
import graficoB
import calculoB
import calculo

def main():
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
    all_data=all_data[all_data[:, -1] <= 7] 
    print("Dados filtrados com sucesso!")
    feats,lables,_=calculo.extrair_features(all_data)
    segs_raw, labels_seg, parts_seg = calculo.segmentation(all_data)
    # =========================================================================
    # 1. DATA AUGMENTATION
    # =========================================================================
    print("INICIANDO DATA AUGMENTATION...\n")
    #1.1 Analise de Balanceamento
    print("Analisando balanceamento dos dados...")
    calculoB.check_balance(lables)
    graficoB.plot_balance(lables, title="Distribuição Original")
    #1.2 Create SMOTE
    print("Aplicando SMOTE para balanceamento dos dados...")
    feats_aug, lables_aug = calculoB.apply_smote(feats, lables)
    calculoB.check_balance(lables_aug)
    graficoB.plot_balance(lables_aug, title="Distribuição Após SMOTE")
    #1.3 Visualização de Augmentação
    print("Visualizando dados após augmentação...")
    graficoB.generate_examples(feats_aug, lables_aug)
    # =========================================================================
    # 2.Embedding features
    # =========================================================================
    print("Extraindo features de embedding...")
    #2.1 Extract embedding(30HZ)
    print("Extraindo features de embedding a 30HZ...")
    embed_feats_30Hz = calculoB.extract_embedding_features(segs_raw, target_fs=30, n_components=64)
    print("Estas são as dimensões das features de embedding a 30HZ:", embed_feats_30Hz.shape)
    #==========================================================================
    #3. Data splitting strategy
    #==========================================================================
    print("Dividindo os dados em treino, validação e teste...")
    #3.1 TVT Split
    print("Dividindo os dados usando a estratégia TVT Split...")
    train_idx, val_idx, test_idx = calculoB.tvt_split(embed_feats_30Hz, labels_seg)
    X_train=embed_feats_30Hz[train_idx]
    y_train=labels_seg[train_idx]
    X_val=embed_feats_30Hz[val_idx]
    X_test=embed_feats_30Hz[test_idx]
    print("Divisão concluída com sucesso!")
    #3.2 Subject Level Split
    print("Dividindo os dados usando a estratégia TVT Split por participante...")
    train_idx, val_idx, test_idx = calculoB.split_between_subjects(parts_seg)
    #3.4 Method: "all";"pca";"relief"
    print("a)Usando método de seleção de features: All features")
    X_train_all, X_val_all, X_test_all, scaler_all, _= calculoB.combined_split(X_train, y_train, train_idx, X_val, X_test, method="all")
    print("b)Usando método de seleção de features: pca features")
    X_train_pca, X_val_pca, X_test_pca, scaler_pca, pca_model = calculoB.combined_split(X_train, y_train, train_idx, X_val, X_test, method="pca")
    print("c)Usando método de seleção de features: relief features")
    X_train_relief, X_val_relief, X_test_relief, scaler_relief, selector_relief = calculoB.combined_split(X_train, y_train, train_idx, X_val, X_test, method="relief")
    print("Seleção de features concluída com sucesso!")
    #==========================================================================
    #4. Model learning
    #==========================================================================
    print("Iniciando o treinamento do modelo...")
    #Implementar a k-Neaarest neighbors (k-NN) para classificação
    

if __name__ == "__main__":
    main()