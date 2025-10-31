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
        for col_idx, name in magnitudes_info.items():
            print(f"A gerar gráfico 2D para {name} com k={k}...")
            graficos.scatter_outliers_zscore(device_data, outliers_mask, k, col_idx, name)
    print("Gráficos gerados com sucesso!")
    # 3.5 - 3.1 VS 3.4
    print("\n--- Requisito 3.5: Comparação Boxplots vs Outliers Z-Score ---")
    print("Os dois métodos discordam drasticamente sobre o que é um outlier, especialmente em atividades de transição e estáticas, porque o Z-Score assume uma distribuição de dados (normal) que os dados reais não seguem. Boxplots fornecem uma visão mais robusta dos dados, enquanto o Z-Score pode ser enganoso em distribuições não normais.")
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
    outliers_dbscan_mask = calculo.aplicar_dbscan(acc_data_3d, eps=0.5, min_samples=75)
    plot_title_outliers_dbscan = f'Outliers DBSCAN para Disp {device_id}, Ativ {activity_id}'
    graficos.visualizar_outliers_3d(acc_data_3d, outliers_dbscan_mask, plot_title_outliers_dbscan)
    print(f"Número de outliers detetados com DBSCAN: {np.sum(outliers_dbscan_mask)}")
    print("\nAnálise concluída.")
    #4
    print("\n" + "="*80)
    print("INICIANDO ANÁLISE DOS REQUISITOS 4.1 a 4.6")
    print("="*80)
    
    # Executar análise completa
    resultados = calculo.executar_analise_completa(all_data_with_magnitudes)
    
    if resultados is None:
        print("\nERRO: Análise não pôde ser completada.")
        return
    
    # ========================================================================
    # VISUALIZAÇÕES ADICIONAIS
    # ========================================================================
    
    print("\n" + "="*80)
    print("VISUALIZAÇÕES ADICIONAIS")
    print("="*80)
    
    # Gráficos PCA
    print("\nA gerar visualizações PCA...")
    graficos.plot_variancia_explicada_pca(resultados['pca'], resultados['pc_75'])
    graficos.plot_pca_2d(resultados['features_pca'], resultados['labels'])
    graficos.plot_pca_3d(resultados['features_pca'], resultados['labels'])
    
    # Gráficos Fisher Score
    print("\nA gerar visualizações Fisher Score...")
    graficos.plot_fisher_scores(resultados['fisher_scores'], top_n=20)
    
    # Gráficos ReliefF
    print("\nA gerar visualizações ReliefF...")
    graficos.plot_relieff_weights(resultados['relieff_weights'], top_n=20)
    
    # Comparação visual
    print("\nA gerar comparação Fisher vs ReliefF...")
    graficos.plot_comparacao_fisher_relieff(
        resultados['fisher_ranking'], 
        resultados['relieff_ranking'], 
        top_n=10
    )
    
    # Matriz de correlação das top features
    print("\nA gerar matriz de correlação...")
    top_10_fisher = resultados['fisher_ranking'][:10]
    graficos.plot_matriz_correlacao_features(
        resultados['feature_matrix'], 
        top_10_fisher,
        ['F'+str(i) for i in top_10_fisher]
    )
    
    # ========================================================================
    # RESUMO FINAL
    # ========================================================================
    
    print("\n" + "="*80)
    print("RESUMO FINAL DA ANÁLISE")
    print("="*80)
    
    print(f"""
📊 ESTATÍSTICAS GERAIS:
  • Total de amostras originais: {all_data.shape[0]}
  • Total de segmentos extraídos: {resultados['feature_matrix'].shape[0]}
  • Dimensionalidade original: {resultados['feature_matrix'].shape[1]} features
  • Atividades únicas: {len(np.unique(resultados['labels']))}
  • Dispositivos únicos: {len(np.unique(resultados['dispositivos']))}

🔬 PCA (Redução de Dimensionalidade):
  • Componentes para 75% variância: {resultados['pc_75']}
  • Redução de dimensionalidade: {(1 - resultados['pc_75']/resultados['feature_matrix'].shape[1])*100:.1f}%
  • Variância explicada PC1: {resultados['pca'].explained_variance_ratio_[0]*100:.2f}%
  • Variância explicada PC2: {resultados['pca'].explained_variance_ratio_[1]*100:.2f}%

🎯 SELEÇÃO DE FEATURES:
  • Top 10 Fisher Score: {list(resultados['fisher_ranking'][:10])}
  • Top 10 ReliefF: {list(resultados['relieff_ranking'][:10])}
  • Features comuns (Top 10): {len(set(resultados['fisher_ranking'][:10]) & set(resultados['relieff_ranking'][:10]))}

💡 RECOMENDAÇÕES:
  1. Para classificação, considerar as Top 10-15 features de Fisher ou ReliefF
  2. PCA com {resultados['pc_75']} componentes mantém 75% da informação
  3. Dados NÃO são normalmente distribuídos → preferir métodos não-paramétricos
  4. Alta variabilidade entre dispositivos → considerar normalização por dispositivo
  5. Atividades de transição têm maior densidade de outliers
    """)
    
    # Guardar resultados (opcional)
    print("\n" + "="*80)
    print("A guardar resultados...")
    print("="*80)
    
    try:
        np.savez('resultados_analise.npz',
                 feature_matrix=resultados['feature_matrix'],
                 labels=resultados['labels'],
                 dispositivos=resultados['dispositivos'],
                 features_pca=resultados['features_pca'],
                 fisher_scores=resultados['fisher_scores'],
                 fisher_ranking=resultados['fisher_ranking'],
                 relieff_weights=resultados['relieff_weights'],
                 relieff_ranking=resultados['relieff_ranking'],
                 pc_75=resultados['pc_75'])
        print("✓ Resultados guardados em 'resultados_analise.npz'")
    except Exception as e:
        print(f"✗ Erro ao guardar resultados: {e}")
    
    print("\n" + "="*80)
    print("ANÁLISE COMPLETA CONCLUÍDA COM SUCESSO!")
    print("="*80)

if __name__ == "__main__":
    main()
