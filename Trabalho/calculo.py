import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from scipy.stats import kstest, f_oneway, kruskal
from scipy.fft import fft
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

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


def testes_significativos(data):
    print("A realizar testes de significância estatística...")
    #Primeiro, começamos for verificar a normalidade da distribuição com o teste Kolmogorov-Smirnov
    dispositivos=sorted(np.unique(data[:,0]))
    atividades=sorted(np.unique(data[:,11]))
    variaveis={
        12: 'Módulo Aceleração',
        13: 'Módulo Giroscópio',
        14: 'Módulo Magnetómetro'}
    for disp in dispositivos:
        print(f"\nA analisar o dispositivo {int(disp)}:")
        
        disp_dados = data[data[:,0]==disp]
        for var_idx, var_nome in variaveis.items():
            print(f"\n Variável: {var_nome}")
            
            dadospor_atividade = []
            atividades_normais=[]
            print("A realizar o teste de Kolmogorov-Smirnov para cada atividade:\n")
            todasNormais=True
            for ativ in atividades:
                dados_ativ= disp_dados[disp_dados[:,11]==ativ, var_idx]
                if len(dados_ativ)>7: #?? não se este número é adequado
                    dados_norm = (dados_ativ - np.mean(dados_ativ)) / np.std(dados_ativ)
                    ks_stat, p_value = kstest(dados_norm, 'norm')
                    is_normal= p_value > 0.05
                    
                    if not is_normal:
                        todasNormais= False
                        
                    dadospor_atividade.append(dados_ativ)
                    atividades_normais.append(int(ativ))
                    
        print("Vamos testar as significâncias para todas as atividades normais:\n")
        
        if len(dadospor_atividade)>=2:
            if todasNormais:
                print("Todas as atividades seguem uma distribuição normal. A realizar ANOVA...\n")
                f_stat, p_value = f_oneway(*dadospor_atividade)
                if p_value <0.05:
                    print("Diferença significativa encontrada entre as atividades (p < 0.05) usando ANOVA.")
                else:
                    print("Nenhuma diferença significativa encontrada entre as atividades (p >= 0.05) usando ANOVA.")
            else:
                print("Nem todas as atividades têm distribuição normal. A realizar o teste de Kruskal-Wallis...\n")
                h_stat, p_value= kruskal(*dadospor_atividade)
                if p_value <0.05:
                    print("Diferença significativa encontrada entre as atividades (p < 0.05) usando Kruskal-Wallis.")
                else:
                    print("Nenhuma diferença significativa encontrada entre as atividades (p >= 0.05) usando Kruskal-Wallis.")
                    
            print("Estatísticas descritivas por atividade:\n")
            print(f"{'Atividade':<10} {'Média':<15} {'Mediana':<15} {'Desvio Padrão':<15}")
            for i, ativ in enumerate(atividades_normais):
                dados = dadospor_atividade[i]
                print(f"{ativ:<10} {np.mean(dados):<15.4f} {np.std(dados):<15.4f} {np.min(dados):<15.4f} {np.max(dados):<15.4f}")
                
        
def features_temporais(segmento):
    mean_val=0
    features= []
    for col in range(1,10):
        dados= segmento[:,col]
        N= len(dados)
        
        features.append(np.mean(dados))
        features.append(np.median(dados))
        features.append(np.std(dados))
        features.append(np.var(dados))
        features.append(np.sqrt(np.mean(dados**2)))

        if N >1:
            features.append(np.mean(np.diff(dados)))
        else:
            features.append(0)

        features.append(skew(dados))
        features.append(kurtosis(dados))
        
        q75,q25= np.percentile(dados, [75 ,25])
        features.append(q75-q25)
        
        if N>1:
            zcr= np.sum(np.diff(np.sign(dados))!=0)/(N-1)
            features.append(zcr)
            
            mean_val= np.mean(dados)
            mcr= np.sum(np.diff(np.sign(dados - mean_val))!=0)/(N-1)
            features.append(mcr)
        else:
            features.append(0)
            features.append(0)
            
    dados_todos_eixos=segmento[:,1:10]
    try:
        coor_matrix_total = np.corrcoef(dados_todos_eixos, rowvar=False)
        coor_matrix =np.nan_to_num(coor_matrix_total)
        indices_superior= np.triu_indices(9, k=1)
        features.extend(coor_matrix[indices_superior])
    except:
        features.extend([0]*36)
        
    return np.array(features)

def features_espectrais(segmento, fs=50):
    features = []
    for col in range (1,10):
        dados = segmento[:,col]
        N= len(dados)
        
        fft_vals = fft(dados)
        fft_mag =np.abs(fft_vals[:N//2])
    
        if np.sum(fft_mag) > 0:
            fft_mag_norm= fft_mag /np.sum(fft_mag)
        else:
            fft_mag_norm= fft_mag
            
        fft_mag_pos = fft_mag_norm[fft_mag_norm >0]
        if len(fft_mag_pos) >0:
            entropia= -np.sum(fft_mag_pos * np.log2(fft_mag_pos))
            features.append(entropia)
        else:
            features.append(0)
    return np.array(features)

def segmentation(data, janela_size=5, overlap= 0.5, fs=50):
    print("A segmentar os dados...\n")
    
    tam_janela= int(janela_size*fs)
    passo= int(tam_janela * (1-overlap))
    
    segmentos= []
    labels= []
    dispositivos= []
    data_sorted= data[data[:,10].argsort()]
    for disp_id in np.unique(data_sorted[:,0]):
        disp_data= data_sorted[data_sorted[:,0]==disp_id]
        i=0
        while i+tam_janela <=len(disp_data):
            segmento= disp_data[i:i+tam_janela]
            atividades_seg = np.unique(segmento[:,11])
            
            if len(atividades_seg)==1:
                segmentos.append(segmento)
                labels.append(atividades_seg[0])
                dispositivos.append(disp_id)
            i += passo
    print(f"Segmentação concluída: {len(segmentos)} segmentos criados.\n")
    return segmentos, np.array(labels), np.array(dispositivos)
        
def extrair_features(data):
    segmentos,labels,dispositivos= segmentation(data)

    if len(segmentos)==0:
        print("Não foi encontrado nenhum segmento!\n")
        return None, None, None
    
    print("A extrair features de cada segmento...\n")
    features_matrix= []
    for i, seg in enumerate(segmentos):
        if i%100==0:
            print(f"Processados {i} de {len(segmentos)} segmentos...")
        feat_temp= features_temporais(seg)
        feat_spec= features_espectrais(seg)
        features_complete= np. concatenate((feat_temp, feat_spec))
        features_matrix.append(features_complete)
        
    return np.array(features_matrix), np.array(labels), np.array(dispositivos)

def apply_pca(features_matrix, n_components=None):
    scaler= StandardScaler()
    features_normalizadas= scaler.fit_transform(features_matrix)
    
    pca = PCA(n_components=n_components)
    features_pca= pca.fit_transform(features_normalizadas)
    
    var_acumulada= 0
    pc_75=None
    
    for i,var in enumerate(pca.explained_variance_ratio_):
        var_acumulada +=var
        print(f"PC{i+1:<4} {var*100:<15.2f}{var_acumulada*100:<15.2f}\n")
        
        if var_acumulada >= 0.75 and pc_75 is None:
            pc_75= i+1
            
    print("Para explicarmos 75% do feature set, devemos utilizar {pc_75} componentes principais.\n")
        
    for i,eigenval in enumerate(pca.explained_variance_[:10]):
        print(f"PC{i+1<5}{eigenval:<15.6f\n}")
            
    return pca, features_pca, scaler, pc_75

def example_pca(pca, features_matrix, scaler,idx_exemplo=0,n_components_75=None):
    features_original= features_matrix[idx_exemplo]
    features_normalizadas= scaler.transform(features_original.reshape(1,-1))
    
    features_pca_full = pca.transform(features_normalizadas)
    
    if n_components_75:
        features_pca_75= features_pca_full[0, :n_components_75]
        print(f"Features PCA (75% variação) para o exemplo {idx_exemplo}:\n{features_pca_75}\n")
        
        
def fisher_score(features, labels):
    n_features= features.shape[1]
    n_classes= len(np.unique(labels))
    fisher_scores= np.zeros(n_features)
    
    mean_global= np.mean(features, axis=0)
    
    for i in range(n_features):
        features_col = features[:,i]
        
        sb=0
        for classe in np.unique(labels):
            indices_classe= labels ==classe
            n_classe= np.sum(indices_classe)
            mean_classe= np.mean(features_col[indices_classe])
            sb += n_classe* (mean_classe - mean_global[i]**2)
            
        sw=0
        for classe in np.unique(labels):
            indices_classe= labels ==classe
            feature_classe=labels ==classe
            sw +=np.sum((feature_classe - np.mean(feature_classe))**2)

        if sw>0:
            fisher_score[i]= sb/sw
        else:
            fisher_score[i]=0
    
    ranking= np.argsort(fisher_scores)[::-1]
    for i in range(min(10,len(ranking))):
        idx= ranking[i]
        print(f"{i+1:<6} {idx:<12}{fisher_scores[idx]:<15.6f}")
    return fisher_scores, ranking

def relieff(features, labels, k=10):
    n_samples, n_features= features.shape
    weights= np.zeros(n_features)
    
    scaler= MinMaxScaler()
    features_norm= scaler.fit_transform(features)
    
    n_iterations = min(n_samples,500)
    indices = np.random.choice(n_samples, n_iterations, replace=False)
    
    for iter_idx, i in enumerate(indices):
        if iter_idx %100 ==0:
            print(f"Processamento da amostra {iter_idx}/{n_iterations}")
        sample= features_norm[i]
        sample_label= labels[i]
        
        distances = np.sqrt(np.sum((features_norm- sample)**2, axis=1))
        distances[i]= np.inf
        
        same_class_mask = labels == sample_label
        same_class_distances = distances.copy()
        same_class_distances[~same_class_mask] = np.inf
        near_hit_indices = np.argsort(same_class_distances)[:k]
        
        # nearMiss: k vizinhos mais próximos de OUTRAS classes
        diff_class_mask = labels != sample_label
        diff_class_distances = distances.copy()
        diff_class_distances[~diff_class_mask] = np.inf
        near_miss_indices = np.argsort(diff_class_distances)[:k]
        
        # Atualizar pesos para cada feature
        for j in range(n_features):
            # Diferença para nearHit (queremos minimizar)
            diff_hit = np.mean(np.abs(sample[j] - features_norm[near_hit_indices, j]))
            
            # Diferença para nearMiss (queremos maximizar)
            diff_miss = np.mean(np.abs(sample[j] - features_norm[near_miss_indices, j]))
            
            # ReliefF weight update
            weights[j] += (diff_miss - diff_hit)
    
    # Normalizar weights
    weights = weights / n_iterations
    
    # Ranking
    ranking = np.argsort(weights)[::-1]
    
    print("\nTop 10 Features (ReliefF):")
    print(f"{'Rank':<6} {'Feature ID':<12} {'Weight':<15}")
    print("-" * 35)
    for i in range(min(10, len(ranking))):
        idx = ranking[i]
        print(f"{i+1:<6} {idx:<12} {weights[idx]:<15.6f}")
    
    return weights, ranking
    
def comparar_fisher_relieff(fisher_scores, fisher_ranking, relieff_weights, relieff_ranking):
    """
    Compara os resultados do Fisher Score e ReliefF.
    """
    print("\n" + "="*80)
    print("REQUISITO 4.6: COMPARAÇÃO FISHER SCORE vs RELIEFF")
    print("="*80)
    
    print("\nTop 10 Features selecionadas por cada método:")
    print(f"{'Rank':<6} {'Fisher Score':<15} {'ReliefF':<15} {'Comum?':<10}")
    print("-" * 50)
    
    top_fisher = set(fisher_ranking[:10])
    top_relieff = set(relieff_ranking[:10])
    comum = top_fisher & top_relieff
    
    for i in range(10):
        f_id = fisher_ranking[i]
        r_id = relieff_ranking[i]
        is_comum = "✓" if (f_id in top_relieff or r_id in top_fisher) else ""
        print(f"{i+1:<6} {f_id:<15} {r_id:<15} {is_comum:<10}")
    
    print(f"\nFeatures em comum no Top 10: {len(comum)}")
    print(f"Features comuns: {sorted(comum)}")
    
    print("\n" + "="*80)
    print("ANÁLISE COMPARATIVA:")
    print("="*80)
    
    print("\nFISHER SCORE:")
    print("  • Método: Estatístico (variância entre/dentro classes)")
    print("  • Tipo: Filtro univariado")
    print("  • Vantagem: Rápido, simples, bom para dados com separação linear")
    print("  • Limitação: Não captura interações entre features")
    print("  • Melhor para: Features com grande separação de médias entre classes")
    
    print("\nRELIEFF:")
    print("  • Método: Baseado em instâncias (vizinhos próximos)")
    print("  • Tipo: Filtro multivariado")
    print("  • Vantagem: Captura dependências entre features, robusto a ruído")
    print("  • Limitação: Mais lento, sensível ao parâmetro k")
    print("  • Melhor para: Features relevantes em contexto (interações)")
    
    print("\nDIFERENÇAS OBSERVADAS:")
    if len(comum) > 7:
        print(f"  • Alta concordância ({len(comum)}/10 features comuns)")
        print("  • Ambos identificam features claramente discriminantes")
    elif len(comum) > 4:
        print(f"  • Concordância moderada ({len(comum)}/10 features comuns)")
        print("  • Alguns critérios diferentes de relevância")
    else:
        print(f"  • Baixa concordância ({len(comum)}/10 features comuns)")
        print("  • Critérios muito diferentes: Fisher prefere separação linear,")
        print("    ReliefF prefere padrões locais de vizinhança")
        
def exemplo_selecao_features(feature_matrix, fisher_ranking, relieff_ranking, idx_exemplo=0):
    """
    Exemplifica a seleção de features para um segmento.
    """
    print(f"\n{'='*80}")
    print(f"REQUISITO 4.6.1: EXEMPLO DE SELEÇÃO DE FEATURES")
    print(f"{'='*80}")
    
    # Selecionar exemplo
    features_original = feature_matrix[idx_exemplo]
    
    print(f"\n1. Features ORIGINAIS (segmento #{idx_exemplo}):")
    print(f"   Total de features: {len(features_original)}")
    print(f"   Primeiras 10: {features_original[:10]}")
    
    # Top 10 Fisher
    top_fisher = fisher_ranking[:10]
    features_fisher = features_original[top_fisher]
    
    print(f"\n2. Features selecionadas por FISHER SCORE (Top 10):")
    print(f"   Índices: {top_fisher}")
    print(f"   Valores: {features_fisher}")
    
    # Top 10 ReliefF
    top_relieff = relieff_ranking[:10]
    features_relieff = features_original[top_relieff]
    
    print(f"\n3. Features selecionadas por RELIEFF (Top 10):")
    print(f"   Índices: {top_relieff}")
    print(f"   Valores: {features_relieff}")
    
    print(f"\n4. REDUÇÃO DE DIMENSIONALIDADE:")
    print(f"   Dimensões originais: {len(features_original)}")
    print(f"   Dimensões após seleção: 10")
    print(f"   Redução: {(1 - 10/len(features_original))*100:.1f}%")
    
def executar_analise_completa(data):
    """
    Executa toda a análise dos requisitos 4.1 a 4.6.
    """
    print("\n" + "="*80)
    print("ANÁLISE COMPLETA - REQUISITOS 4.1 a 4.6")
    print("="*80)
    
    # 4.1 - Testes de significância
    testes_significativos(data)
    
    # 4.2 - Extração de features
    feature_matrix, labels, dispositivos = extrair_features(data)
    
    if feature_matrix is None:
        print("\nERRO: Não foi possível extrair features. Encerrando análise.")
        return
    
    # 4.3 e 4.4 - PCA
    pca, features_pca, scaler, pc_75 =apply_pca(feature_matrix)
    
    # 4.4.1 - Exemplo transformação PCA
    example_pca(pca, scaler, feature_matrix, idx_exemplo=0, n_components_75=pc_75)
    
    # 4.5 - Fisher Score
    fisher_scores, fisher_ranking = fisher_score(feature_matrix, labels)
    
    # 4.5 - ReliefF
    relieff_weights, relieff_ranking = relieff(feature_matrix, labels, k=10)
    
    # 4.6 - Comparação
    comparar_fisher_relieff(fisher_scores, fisher_ranking, relieff_weights, relieff_ranking)
    
    # 4.6.1 - Exemplo seleção
    exemplo_selecao_features(feature_matrix, fisher_ranking, relieff_ranking, idx_exemplo=0)
    print("\n" + "="*80)
    print("ANÁLISE COMPLETA CONCLUÍDA!")
    print("="*80)
    
    return {
        'feature_matrix': feature_matrix,
        'labels': labels,
        'dispositivos': dispositivos,
        'pca': pca,
        'features_pca': features_pca,
        'scaler': scaler,
        'pc_75': pc_75,
        'fisher_scores': fisher_scores,
        'fisher_ranking': fisher_ranking,
        'relieff_weights': relieff_weights,
        'relieff_ranking': relieff_ranking
    }







