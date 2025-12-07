#Meta 2 do Projeto

import os
import openfile
import graficoB
import calculoB
import calculo
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

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
    #5. Evaluation - Report Results
    #=========================================================================
    print("\n" + "="*80)
    print("INICIANDO AVALIAÇÃO COMPLETA (REPORT RESULTS)...")
    print("Com múltiplas iterações para análise estatística robusta")
    print("="*80)
    results, predictions = calculoB.report_results(
        dataset, 
        participants=parts_seg, 
        n_iterations=10  
    )
    print("\nAvaliação concluída com sucesso!")
    
    # 5.3 - Análise Comparativa dos Resultados (com Teste de Friedman e Student Pareado)
    print("\n" + "="*80)
    print("INICIANDO ANÁLISE COMPARATIVA (REQUISITO 5.3)...")
    print("Inclui: Teste de Friedman + Teste T de Student Pareado")
    print("="*80)
    calculoB.analyze_results_5_3(results, predictions)
    
    #=========================================================================
    #6. Deployment
    #=========================================================================
    print("\n" + "="*80)
    print("REQUISITO 6: DEPLOYMENT DO MODELO")
    print("="*80)
    
    # 6.1 Selecionar a melhor configuração baseada nos resultados (5.3)
    print("\n6.1. SELECIONANDO MELHOR MODELO PARA DEPLOYMENT...")
    print("-" * 70)
    best_config_name = max(results.items(), key=lambda x: x[1]['mean_acc'])[0]
    best_config_metrics = results[best_config_name]
    
    print(f"\nMelhor Configuração: {best_config_name}")
    print(f"  Acurácia Média: {best_config_metrics['mean_acc']:.4f}")
    print(f"  Desvio Padrão: {best_config_metrics['std_acc']:.4f}")
    print(f"  Precision: {best_config_metrics['metrics'][0]['precision']:.4f}")
    print(f"  Recall: {best_config_metrics['metrics'][0]['recall']:.4f}")
    print(f"  F1-Score: {best_config_metrics['metrics'][0]['f1_score']:.4f}")
    
    # 6.2 Extrair parâmetros e modelo do Requisito 5
    print("\n6.2. EXTRAINDO MODELO TREINADO DO REQUISITO 5...")
    print("-" * 70)
    
    # Extrair parâmetros da melhor configuração
    parts = best_config_name.split('_')
    best_dataset = parts[0] + '_' + parts[1]  # Dataset_A ou Dataset_B
    best_method = parts[2]  # all, pca, relief
    best_strategy = '_'.join(parts[3:])  # Within-Subject ou Between-Subject
    
    print(f"\nParâmetros do Modelo Selecionado:")
    print(f"  Dataset: {best_dataset}")
    print(f"  Método de Seleção: {best_method}")
    print(f"  Estratégia de Split: {best_strategy}")
    
    # 6.3 Recuperar modelo, scaler e selector do Requisito 5
    print("\n6.3. RECUPERANDO COMPONENTES DO MODELO DO REQUISITO 5...")
    print("-" * 70)
    
    # O modelo, scaler e selector já foram treinados no Requisito 5
    # Estão armazenados nos dados dos configs do Requisito 5
    # Aqui reconstruímos o melhor modelo a partir dos resultados
    
    # Selecionar dataset apropriado
    if best_dataset == "Dataset_A":
        X_train = X_train_manual
        y_train = y_train_manual
        X_val = X_val_manual
        y_val = y_val_manual
        X_test = X_test_manual
        y_test = y_test_manual
    else:  # Dataset_B
        X_train = X_train_seg
        y_train = y_train_seg
        X_val = X_val_seg
        y_val = y_val_seg
        X_test = X_test_seg
        y_test = y_test_seg
    
    # Aplicar feature selection do Requisito 5 para reconstruir o modelo
    X_train_fs, X_val_fs, X_test_fs, scaler_model, selector_model = calculoB.combined_split(
        X_train, y_train, X_val, X_test, method=best_method
    )
    
    # Obter o melhor k do predictions (primeira iteração)
    best_k_model = predictions[best_config_name]['best_k']
    
    # Retreinar o modelo com o melhor k usando train + val (padrão do Requisito 5)
    X_train_val_combined = np.vstack((X_train_fs, X_val_fs))
    y_train_val_combined = np.hstack((y_train, y_val))
    
    print(f"\nReconstruindo modelo com melhores parâmetros do Requisito 5...")
    print(f"  Melhor k encontrado: {best_k_model}")
    print(f"  Dados: {X_train_val_combined.shape[0]} amostras combinadas (train+val)")
    
    # Criar e treinar modelo final
    from sklearn.neighbors import KNeighborsClassifier
    model_deployed = KNeighborsClassifier(n_neighbors=best_k_model)
    model_deployed.fit(X_train_val_combined, y_train_val_combined)
    
    # Avaliar no conjunto de teste
    y_test_pred = model_deployed.predict(X_test_fs)
    test_accuracy_deployed = np.mean(y_test_pred == y_test)
    
    print(f"\nModelo Reconstruído:")
    print(f"  k-NN com k={best_k_model}")
    print(f"  Acurácia no Teste: {test_accuracy_deployed:.4f}")
    
    # 6.4 Criar estrutura de deployment
    print("\n6.4. CRIANDO PIPELINE DE DEPLOYMENT...")
    print("-" * 70)
    
    # Determinar tipo de extrator de features
    if best_dataset == "Dataset_B":
        feature_extractor_type = 'embedding'
    else:
        feature_extractor_type = 'manual'
    
    # Estrutura com todos os componentes necessários para deployment
    deployment_model = {
        'config': {
            'dataset': best_dataset,
            'method': best_method,
            'strategy': best_strategy,
            'feature_extractor': feature_extractor_type,
            'k_value': best_k_model,
            'sample_rate': 51.5  # Hz
        },
        'model': model_deployed,
        'scaler': scaler_model,
        'selector': selector_model,
        'selector_type': best_method if best_method != 'all' else None,
        'metrics': best_config_metrics['metrics'][0],
        'best_accuracy': best_config_metrics['mean_acc']
    }
    
    print("\nComponentes de Deployment:")
    print(f"  ✓ Modelo k-NN (k={best_k_model})")
    print(f"  ✓ Scaler ({type(scaler_model).__name__})")
    print(f"  ✓ Feature Selector ({best_method})")
    print(f"  ✓ Configuração: {best_dataset} - {best_method} - {best_strategy}")
    
    # 6.5 Testar função de predição
    print("\n6.5. TESTANDO FUNÇÃO DE DEPLOYMENT...")
    print("-" * 70)
    
    # Testar com alguns exemplos de teste
    n_test_samples = min(3, len(X_test_fs))
    print(f"\nTestando modelo deployado com {n_test_samples} amostras:")
    
    for i in range(n_test_samples):
        # Usar dados já processados pelo feature selector
        X_sample = X_test_fs[i:i+1]
        y_true_sample = y_test[i]
        
        # Fazer predição
        prediction = model_deployed.predict(X_sample)[0]
        probas = model_deployed.predict_proba(X_sample)[0]
        confidence = np.max(probas)
        
        print(f"\n  Amostra {i+1}:")
        print(f"    Atividade Predita: {int(prediction)}")
        print(f"    Confiança: {confidence:.4f}")
        print(f"    Verdadeira: {int(y_true_sample)}")
        print(f"    {'✓ CORRETO' if int(prediction) == int(y_true_sample) else '✗ INCORRETO'}")
    
    # 6.6 Salvar informações de deployment
    print("\n6.6. INFORMAÇÕES DE DEPLOYMENT...")
    print("-" * 70)
    print(f"\nModelo de Deployment Criado com Sucesso!")
    print(f"  Fonte: Melhor modelo do Requisito 5")
    print(f"  Dataset Utilizado: {deployment_model['config']['dataset']}")
    print(f"  Método de Seleção: {deployment_model['config']['method']}")
    print(f"  Estratégia: {deployment_model['config']['strategy']}")
    print(f"  Acurácia Média (Requisito 5): {deployment_model['best_accuracy']:.4f}")
    print(f"  Acurácia em Teste (Reconstruído): {test_accuracy_deployed:.4f}")
    print(f"\nO modelo está pronto para ser utilizado em produção!")
    print(f"Utilize a função deploy_model() de calculoB.py com os parâmetros acima.")
    
    #=========================================================================
    #7. Go further
    #=========================================================================
    
    

if __name__ == "__main__":
    main() 