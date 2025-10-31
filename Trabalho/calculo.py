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

#Testes signitificativos para a alínea 4.1. 
#Estes testes ajudam-nos a determinar se as diferenças que encontramos entre os valores
#das variáveis nas diferentes atividades se devem a algum erro nos dados ou a um valor específico.
#Primeiro, começamos por verificar a normalidade dos dados com o teste indicado pelo professor (Kolmogorov-Smirnov).
#De seguida, dependendo do resultado, vamos aplicar um teste estatístico específico.

def testes_significativos(data):
    print("A realizar testes de significância estatística...")
    #Primeiro, começamos for verificar a normalidade dos dados com o teste Kolmogorov-Smirnov
    dispositivos=sorted(np.unique(data[:,0])) #vamos buscar os valores únicos à coluna0, que contém os dispositivos
    atividades=sorted(np.unique(data[:,11]))#organizamos os dads em ordem ascendente
    variaveis={ #criamos um dicionário para facilitar o mapeamento 
        12: 'Módulo Aceleração',
        13: 'Módulo Giroscópio',
        14: 'Módulo Magnetómetro'}
    for disp in dispositivos: #vamos percorrer cada dispositivo ID
        print(f"\nA analisar o dispositivo {int(disp)}:")
        
        disp_dados = data[data[:,0]==disp] #filtramos o array original para termos os dados de apenas um dispositivo
        for var_idx, var_nome in variaveis.items(): #vamos percorrer cad avariável do dicionário
        #var_idx: índice da coluna; var_nome: nome da variável
            print(f"\n Variável: {var_nome}")
            
            dadospor_atividade = [] #vamos guardar os dados de cada atividade aqui
            atividades_normais=[] #vamos guardar as atividades com dados relevantes
            print("A realizar o teste de Kolmogorov-Smirnov para cada atividade:\n")
            todasNormais=True #assumimos que os dados têm uma distribuição normal
            for ativ in atividades: #vamos percorrer cada atividade
                dados_ativ= disp_dados[disp_dados[:,11]==ativ, var_idx] #filtramos os dados para a atividade específica
                if len(dados_ativ)>2: #Verificamos se temos dados suficientes para o teste
                    # Normalizar dados para o teste KS
                    if np.std(dados_ativ) > 0:
                        dados_norm = (dados_ativ - np.mean(dados_ativ)) / np.std(dados_ativ) 
                        #Temos de alterar a mean e std dos nossos dados 
                        #para 0 e 1, para conseguirmos fazer uma comparação com o standart
                        #o teste KS compara a distribuição empírica dos dados com uma distribuição 
                        #normal teórica.
                        ks_stat, p_value = kstest(dados_norm, 'norm') #teste KS
                    else:
                        p_value = 1.0 # Dados constantes, tecnicamente não violam a normalidade
                        
                    is_normal= p_value > 0.05 #os dados são normais, a hipótese H0 é aceite
                    
                    if not is_normal:
                        todasNormais= False #se a condição H0 falhar, os dados não seguem uma distribuição normal
                        
                    dadospor_atividade.append(dados_ativ) #guardamos os dados da atividade atual
                    atividades_normais.append(int(ativ)) #guardamos os labels das atividades
            
            # Verificar se temos dados suficientes para os testes
            if len(dadospor_atividade) < 2:
                print("Não há dados de atividade suficientes para comparação estatística.")
                continue
                
            print("Vamos testar as significâncias para todas as atividades válidas:\n")

            #Se os dados tiverem uma distribuição normal, utiliza-se o teste estatístico ANOVA
            if todasNormais: 
                print("Todas as atividades seguem uma distribuição normal. A realizar ANOVA...\n")
                f_stat, p_value = f_oneway(*dadospor_atividade) #realizamos o teste ANOVA
                #a sintaxe *dadospor_atividade serve para desempacotar a lista de arrays
                #O objetivo deste teste é comparar as médias das diferentes atividades
                if p_value <0.05:
                    print(f"Diferença significativa encontrada (p={p_value:.4e}) usando ANOVA.")
                else:
                    print(f"Nenhuma diferença significativa encontrada (p={p_value:.4f}) usando ANOVA.")
                    
            else:  
                print("Nem todas as atividades têm distribuição normal. A realizar o teste de Kruskal-Wallis...\n")
                h_stat, p_value= kruskal(*dadospor_atividade)
                if p_value <0.05:
                    print(f"Diferença significativa encontrada (p={p_value:.4e}) usando Kruskal-Wallis.")
                else:
                    print(f"Nenhuma diferença significativa encontrada (p={p_value:.4f}) usando Kruskal-Wallis.")
                    
            print("Estatísticas descritivas por atividade:\n")
            print(f"{'Atividade':<10} {'Média':<15} {'Desvio Padrão':<15} {'Mediana':<15}")
            for i, ativ in enumerate(atividades_normais):
                dados = dadospor_atividade[i]
                if len(dados) > 0:
                    print(f"{ativ:<10} {np.mean(dados):<15.4f} {np.std(dados):<15.4f} {np.median(dados):<15.4f}")
                else:
                    print(f"{ativ:<10} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
                
        
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
        
        if N == 0:
            features.append(0)
            continue
            
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
    
    print("\n" + "="*80)
    print("REQUISITO 4.4: ANÁLISE DE COMPONENTES PRINCIPAIS (PCA)")
    print("="*80)
    
    var_acumulada= 0
    pc_75=None
    
    print(f"{'Componente':<12} {'Variância (%)':<15} {'Variância Acumulada (%)':<15}")
    for i,var in enumerate(pca.explained_variance_ratio_):
        var_acumulada +=var
        if i < 15 or (i+1) % 10 == 0: # Imprimir os 15 primeiros e depois de 10 em 10
            print(f"PC{i+1:<10} {var*100:<15.2f}{var_acumulada*100:<15.2f}")
        
        if var_acumulada >= 0.75 and pc_75 is None:
            pc_75= i+1
            print("-" * 55)
            print(f"PC{i+1:<10} {var*100:<15.2f}{var_acumulada*100:<15.2f} <-- Limiar de 75% atingido")
            print("-" * 55)
            
    
    print(f"\nPara explicarmos 75% do feature set, devemos utilizar {pc_75} componentes principais.\n")
        
    print("Top 10 Eigenvalues (Importância da Componente):")
    for i,eigenval in enumerate(pca.explained_variance_[:10]):
        print(f"PC{i+1:<5} {eigenval:<15.6f}")
            
    return pca, features_pca, scaler, pc_75

def example_pca(pca, features_matrix, scaler,idx_exemplo=0,n_components_75=None):
    print("\n" + "="*80)
    print("REQUISITO 4.4.1: EXEMPLO DE TRANSFORMAÇÃO PCA")
    print("="*80)
    
    features_original= features_matrix[idx_exemplo]
    features_normalizadas= scaler.transform(features_original.reshape(1,-1))
    
    features_pca_full = pca.transform(features_normalizadas)
    
    print(f"Features originais (segmento #{idx_exemplo}, 10 primeiras):")
    print(f"  {features_original[:10]}")
    
    print(f"\nFeatures normalizadas (segmento #{idx_exemplo}, 10 primeiras):")
    print(f"  {features_normalizadas[0, :10]}")
    
    print(f"\nFeatures PCA (Full) (segmento #{idx_exemplo}, 10 primeiras):")
    print(f"  {features_pca_full[0, :10]}")
    
    if n_components_75:
        features_pca_75= features_pca_full[0, :n_components_75]
        print(f"\nFeatures PCA (reduzidas para 75% variação) para o exemplo {idx_exemplo}:")
        print(f"  Dimensões: {len(features_pca_75)}")
        print(f"  Valores (primeiras 10): {features_pca_75[:10]}\n")
        
# --- NOVA FUNÇÃO ADICIONADA ---
def print_pca_analysis():
    """
    Imprime a análise de vantagens e limitações do PCA (Requisito 4.4.2)
    """
    print("\n" + "="*80)
    print("REQUISITO 4.4.2: VANTAGENS E LIMITAÇÕES DO PCA")
    print("="*80)
    
    print("\n💡 VANTAGENS:")
    print("  • Redução de Dimensionalidade: Comprime um grande número de features (como as "
          f"{'108'} que extraímos) num conjunto muito menor (ex: {_pc_75_placeholder} componentes), "
          "mantendo a maior parte da variância (informação).")
    print("  • Remoção de Redundância: O PCA cria componentes não correlacionados, "
          "eliminando a multicolinearidade entre as features originais (ex: correlações entre "
          "média e mediana).")
    print("  • Visualização: Permite visualizar datasets de alta dimensão em 2D ou 3D "
          "(usando PC1, PC2, PC3), ajudando a identificar clusters ou padrões.")
    print("  • Performance: Modelos de Machine Learning treinam mais rapidamente e podem "
          "ter melhor generalização (reduzindo overfitting) com menos features.")

    print("\n⚠️ LIMITAÇÕES:")
    print("  • Perda de Interpretabilidade: As Componentes Principais (PCs) são "
          "combinações lineares de *todas* as features originais. Perdemos a capacidade "
          "de dizer 'a média da aceleração foi importante'. Em vez disso, dizemos 'a PC1 foi "
          "importante'.")
    print("  • Sensibilidade ao Escalamento: O PCA é muito sensível à escala das features. "
          "(NOTA: Nós mitigámos isto ao usar o `StandardScaler` antes de aplicar o PCA).")
    print("  • Suposição de Linearidade: O PCA assume que as relações entre as features "
          "são lineares. Pode falhar em capturar padrões complexos e não-lineares.")
    print("  • Variância vs. Importância: O PCA assume que as direções de maior variância "
          "são as mais importantes para a classificação, o que nem sempre é verdade.")
# --- FIM DA NOVA FUNÇÃO ---

def fisher_score(features, labels):
    print("\n" + "="*80)
    print("REQUISITO 4.5: SELEÇÃO DE FEATURES (FISHER SCORE)")
    print("="*80)
    
    n_features= features.shape[1]
    n_classes= len(np.unique(labels))
    fisher_scores= np.zeros(n_features) # <-- Esta é a variável correta (plural)
    
    mean_global= np.mean(features, axis=0)
    
    for i in range(n_features):
        features_col = features[:,i]
        
        sb=0 # Variância inter-classes (Between)
        mean_classe_geral = {} # Armazenar médias para o cálculo de sw
        
        for classe in np.unique(labels):
            indices_classe= labels ==classe
            n_classe= np.sum(indices_classe)
            if n_classe > 0:
                mean_classe= np.mean(features_col[indices_classe])
                mean_classe_geral[classe] = mean_classe
                sb += n_classe* (mean_classe - mean_global[i])**2
            
        sw=0 # Variância intra-classes (Within)
        for classe in np.unique(labels):
            indices_classe= labels ==classe
            if np.sum(indices_classe) > 0:
                feature_classe= features_col[indices_classe]
                mean_classe = mean_classe_geral[classe] # Usar a média calculada
                sw +=np.sum((feature_classe - mean_classe)**2)

        if sw>0:
            # --- CORREÇÃO DO ERRO ---
            fisher_scores[i]= sb/sw  # Era: fisher_score[i]
        else:
            # --- CORREÇÃO DO ERRO ---
            fisher_scores[i]=0       # Era: fisher_score[i]
    
    ranking= np.argsort(fisher_scores)[::-1]
    
    print("\nTop 10 Features (Fisher Score):")
    print(f"{'Rank':<6} {'Feature ID':<12} {'Score':<15}")
    print("-" * 35)
    for i in range(min(10,len(ranking))):
        idx= ranking[i]
        print(f"{i+1:<6} {idx:<12}{fisher_scores[idx]:<15.6f}")
        
    return fisher_scores, ranking

def relieff(features, labels, k=10):
    print("\n" + "="*80)
    print("REQUISITO 4.5: SELEÇÃO DE FEATURES (RELIEFF)")
    print("="*80)
    
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
        # Garantir que temos vizinhos suficientes
        if np.sum(same_class_mask) > 1:
            near_hit_indices = np.argsort(same_class_distances)[:k]
        else:
            near_hit_indices = []

        
        # nearMiss: k vizinhos mais próximos de OUTRAS classes
        diff_class_mask = labels != sample_label
        diff_class_distances = distances.copy()
        diff_class_distances[~diff_class_mask] = np.inf
        if np.sum(diff_class_mask) > 0:
            near_miss_indices = np.argsort(diff_class_distances)[:k]
        else:
            near_miss_indices = []

        
        # Atualizar pesos para cada feature
        for j in range(n_features):
            # Diferença para nearHit (queremos minimizar)
            if len(near_hit_indices) > 0:
                diff_hit = np.mean(np.abs(sample[j] - features_norm[near_hit_indices, j]))
            else:
                diff_hit = 0
            
            # Diferença para nearMiss (queremos maximizar)
            if len(near_miss_indices) > 0:
                diff_miss = np.mean(np.abs(sample[j] - features_norm[near_miss_indices, j]))
            else:
                diff_miss = 0
            
            # ReliefF weight update
            weights[j] += (diff_miss - diff_hit)
    
    # Normalizar weights
    if n_iterations > 0:
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
    print(f"{'Rank':<6} {'Fisher Score (ID)':<20} {'ReliefF (ID)':<20}")
    print("-" * 50)
    
    top_fisher = set(fisher_ranking[:10])
    top_relieff = set(relieff_ranking[:10])
    comum = top_fisher & top_relieff
    
    for i in range(10):
        f_id = fisher_ranking[i]
        r_id = relieff_ranking[i]
        print(f"{i+1:<6} {f_id:<20} {r_id:<20}")
    
    print(f"\nFeatures em comum no Top 10: {len(comum)}")
    print(f"Features comuns: {sorted(list(comum))}")
    
    print("\n" + "="*80)
    print("ANÁLISE COMPARATIVA:")
    print("="*80)
    
    print("\nFISHER SCORE:")
    print("  • Método: Estatístico (variância inter/intra-classes).")
    print("  • Tipo: Filtro univariado (analisa cada feature isoladamente).")
    print("  • Vantagem: Muito rápido e simples de calcular.")
    print("  • Limitação: Ignora interações e redundâncias entre features. "
          "Pode selecionar features redundantes (ex: média e mediana, que são "
          "altamente correlacionadas).")
    
    print("\nRELIEFF:")
    print("  • Método: Baseado em instâncias (vizinhos próximos).")
    print("  • Tipo: Filtro multivariado (sensível ao contexto das features).")
    print("  • Vantagem: Consegue detetar features que só são úteis em conjunto "
          "(interações) e penaliza features redundantes.")
    print("  • Limitação: Computacionalmente mais caro (precisa de calcular "
          "distâncias) e sensível ao parâmetro 'k' (número de vizinhos).")
    
    print("\nDIFERENÇAS OBSERVADAS:")
    if len(comum) > 7:
        print(f"  • Alta concordância ({len(comum)}/10). Isto sugere que as features mais "
              "discriminantes são tão fortes que ambos os métodos as detetam, "
              "independentemente das suas interações.")
    elif len(comum) > 4:
        print(f"  • Concordância moderada ({len(comum)}/10). Ambos os métodos encontram "
              "um núcleo de features úteis, mas diferem nas restantes, "
              "provavelmente porque o ReliefF encontrou interações que o Fisher ignorou.")
    else:
        print(f"  • Baixa concordância ({len(comum)}/10). Isto indica que os dois métodos "
              "têm visões muito diferentes. O Fisher escolhe features com "
              "médias de classe muito separadas, enquanto o ReliefF "
              "escolhe features que ajudam a separar vizinhos difíceis.")
        
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
    print(f"   Valores (das features originais nesses índices): {features_fisher}")
    
    # Top 10 ReliefF
    top_relieff = relieff_ranking[:10]
    features_relieff = features_original[top_relieff]
    
    print(f"\n3. Features selecionadas por RELIEFF (Top 10):")
    print(f"   Índices: {top_relieff}")
    print(f"   Valores (das features originais nesses índices): {features_relieff}")
    
    print(f"\n4. REDUÇÃO DE DIMENSIONALIDADE:")
    print(f"   Dimensões originais: {len(features_original)}")
    print(f"   Dimensões após seleção: 10")
    print(f"   Redução: {(1 - 10/len(features_original))*100:.1f}%")

# --- NOVA FUNÇÃO ADICIONADA ---
def print_selection_analysis():
    """
    Imprime a análise de vantagens e limitações da Seleção de Features (Req 4.6.2)
    """
    global _pc_75_placeholder # Usar a variável global para a string
    print(f"\n{'='*80}")
    print(f"REQUISITO 4.6.2: VANTAGENS E LIMITAÇÕES DA SELEÇÃO DE FEATURES")
    print(f"{'='*80}")
    
    print("\n💡 VANTAGENS:")
    print("  • Interpretabilidade: Esta é a maior vantagem sobre o PCA. O modelo final "
          "usa as features originais (ex: 'média do giroscópio-x', 'std da aceleração-z'). "
          "Podemos interpretar *quais* características físicas são importantes.")
    print("  • Eficiência: Criar um modelo com 10 features é muito mais rápido do que "
          "com 108 (ou mesmo com as "
          f"{_pc_75_placeholder} do PCA).")
    print("  • Robustez a Overfitting: Ao remover features irrelevantes ou redundantes, "
          "o modelo pode generalizar melhor para novos dados.")

    print("\n⚠️ LIMITAÇÕES:")
    print("  • Perda de Informação: Ao contrário do PCA, que *comprime* a informação, "
          "a seleção *descarta* features. Se uma feature tiver uma pequena contribuição "
          "mas for útil em conjunto com outras, pode ser descartada, perdendo-se essa "
          "informação.")
    print("  • Risco de Sub-otimização: Métodos de filtro (como Fisher e ReliefF) "
          "selecionam features *antes* do treino do modelo. Eles podem não escolher o "
          "conjunto de features que seria ótimo *para esse modelo específico* "
          "(ex: um SVM ou uma Random Forest).")
    print("  • Dependência do Método: Como vimos, Fisher (univariado) e ReliefF "
          "(multivariado) podem dar resultados diferentes. A escolha do método de "
          "seleção influencia muito o resultado.")
# --- FIM DA NOVA FUNÇÃO ---

# Variável global para passar o n_componentes para os prints
_pc_75_placeholder = 'N/A'

def executar_analise_completa(data):
    """
    Executa toda a análise dos requisitos 4.1 a 4.6.
    """
    global _pc_75_placeholder # Declarar que vamos modificar a variável global
    
    print("\n" + "="*80)
    print("ANÁLISE COMPLETA - REQUISITOS 4.1 a 4.6")
    print("="*80)
    
    # 4.1 - Testes de significância
    testes_significativos(data)
    
    # 4.2 - Extração de features
    feature_matrix, labels, dispositivos = extrair_features(data)
    
    if feature_matrix is None or feature_matrix.size == 0:
        print("\nERRO: Não foi possível extrair features. Encerrando análise.")
        return None
    
    if feature_matrix.shape[0] != labels.shape[0]:
        print("\nERRO: Inconsistência entre número de features e labels.")
        return None
        
    print(f"Matriz de features criada com sucesso: {feature_matrix.shape}")
    
    # 4.3 e 4.4 - PCA
    pca, features_pca, scaler, pc_75 = apply_pca(feature_matrix)
    _pc_75_placeholder = str(pc_75) # Atualizar a variável global
    
    # 4.4.1 - Exemplo transformação PCA
    example_pca(pca, feature_matrix, scaler, idx_exemplo=0, n_components_75=pc_75)
    
    # --- CHAMADA À NOVA FUNÇÃO ---
    print_pca_analysis()
    
    # 4.5 - Fisher Score
    fisher_scores, fisher_ranking = fisher_score(feature_matrix, labels)
    
    # 4.5 - ReliefF
    relieff_weights, relieff_ranking = relieff(feature_matrix, labels, k=10)
    
    # 4.6 - Comparação
    comparar_fisher_relieff(fisher_scores, fisher_ranking, relieff_weights, relieff_ranking)
    
    # 4.6.1 - Exemplo seleção
    exemplo_selecao_features(feature_matrix, fisher_ranking, relieff_ranking, idx_exemplo=0)
    
    # --- CHAMADA À NOVA FUNÇÃO ---
    print_selection_analysis()
    
    print("\n" + "="*80)
    print("ANÁLISE DA SECÇÃO 4 CONCLUÍDA!")
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