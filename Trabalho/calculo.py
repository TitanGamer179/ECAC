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
    #Os objetivos destes testes é comparar as médias das diferentes atividades
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
                if p_value <0.05:
                    print(f"Diferença significativa encontrada (p={p_value:.4e}) usando ANOVA.")
                else:
                    print(f"Nenhuma diferença significativa encontrada (p={p_value:.4f}) usando ANOVA.")
                   
            #Caso não tenham distribuição normal, utiliza-se o teste de Kruskal-Wallis 
            else:  
                print("Nem todas as atividades têm distribuição normal. A realizar o teste de Kruskal-Wallis...\n")
                h_stat, p_value= kruskal(*dadospor_atividade) #realizamos o teste de Kruskal-Wallis
                if p_value <0.05:
                    print(f"Diferença significativa encontrada (p={p_value:.4e}) usando Kruskal-Wallis.")
                else:
                    print(f"Nenhuma diferença significativa encontrada (p={p_value:.4f}) usando Kruskal-Wallis.")
                    
            print("Estatísticas descritivas por atividade:\n")
            print(f"{'Atividade':<10} {'Média':<15} {'Desvio Padrão':<15} {'Mediana':<15}")
            for i, ativ in enumerate(atividades_normais): #utilizamos o enumerate para obter o indice e id da atividade
                dados = dadospor_atividade[i] #vamos buscar o conjunto de dados que lhe compete.
                if len(dados) > 0:
                    print(f"{ativ:<10} {np.mean(dados):<15.4f} {np.std(dados):<15.4f} {np.median(dados):<15.4f}")
                else:
                    print(f"{ativ:<10} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
                
#Desenvolvimento de rotinas à extração de features no exercício 4.2
#Este código tem como objetivo utilizar uma segmentação específica para a extração
#de algumas features identificadas no artigo dado pelo professor.
     
def features_temporais(segmento):
    #Esta função extraí as features temporais identificadas, sendo estas:
    #Mean, median, Standart Deviation, Variance, Root Mean Square Averaged derivatives,
    #Skewness, Kurtosis, Interquartile Range, Zero Crossing Rate, Mean Crossing Rate
    #Pairwise correlation
    mean_val=0
    features= []
    for col in range(1,10): #vamos percorrer os eixos dos sensores
        dados= segmento[:,col] #extraímos todos os dados da coluna
        N= len(dados)
        
        if N == 0:
            features.append(0)
            continue
        
        features.append(np.mean(dados)) #valor médio do sinal
        features.append(np.median(dados)) #Mediana do sinal
        features.append(np.std(dados))#Desvio padrão do sinal
        features.append(np.var(dados)) #Variância do sinal
        features.append(np.sqrt(np.mean(dados**2))) #Raíz quadrada da média dos quadrados

        if N >1:
            features.append(np.mean(np.diff(dados))) #Calcula a diferença entre amostras consecuticas
        else:
            features.append(0)

        features.append(skew(dados)) #Mede a assimetria da distribuição dos dados. 0 se perfeitamente simétrico
        features.append(kurtosis(dados)) # Mede a frequència de valores extremos em comparação com uma distribuição normal
        
        q75,q25= np.percentile(dados, [75 ,25]) #Calcula o 75 e 25 percentil
        features.append(q75-q25) #Medida robusta da dispersão de dados
        
        if N>1:
            zcr= np.sum(np.diff(np.sign(dados))!=0)/(N-1) #Contar o número de vezes que o sinal cruza o eixo zero e normaliza a contagem pelo comprimento do sinal
            #Normalizamos por N-1 PARA OBTER SEMPRE UM VALOR ENTRE 0 E 1
            features.append(zcr)
            
            mean_val= np.mean(dados) #Média do sinal
            mcr= np.sum(np.diff(np.sign(dados - mean_val))!=0)/(N-1) #semelhante ao ZCR mas começa por subtrair a média, e conta quantas vezes o sinal se cruza com o próprio valor médio
            features.append(mcr)
        else: #se tivermos apenas um valor, não conseguimos calcular estas features
            features.append(0)
            features.append(0)
            
    dados_todos_eixos=segmento[:,1:10] #selecionamos todos os 9 eixos de dados
    try:
        coor_matrix_total = np.corrcoef(dados_todos_eixos, rowvar=False) #coeficiente da correlação de Pearson
        coor_matrix =np.nan_to_num(coor_matrix_total) #se a matrix tiver valores conctantes / NaN,
            #substituímos por zeros para evitar erros
        indices_superior= np.triu_indices(9, k=1) #evita pegarmos nos mesmos valores mais do que uma vez
        features.extend(coor_matrix[indices_superior]) #anexamos os valores de coorelação válidos
    except:
        features.extend([0]*36)
        
    return np.array(features) #convertemos a lista final e devolve

def features_espectrais(segmento, fs=51.2):
    #Função muito semelhante à anterior, mas desta vez vamos extrair features
    #espectrais, sendo estas a Entropia Espectral
    features = []
    for col in range (1,10): #voltamos a ver os 9 eixos dos sensores
        dados = segmento[:,col]
        N= len(dados)
        
        if N == 0:
            features.append(0)
            continue
            
        fft_vals = fft(dados) #FFT, converte o sinal do domínio temporal para o domínio da frequência.
        fft_mag =np.abs(fft_vals[:N//2]) #calcula-se a magnitude de cada número, utilizando apenas a primeira metade dos valores
    
        if np.sum(fft_mag) > 0:
            fft_mag_norm= fft_mag /np.sum(fft_mag) #Normalizamos o espectro de magnitude dividindo cada valor pela soma total, fazendo assim com que a soma de todos os valores seja 1.
        else:
            fft_mag_norm= fft_mag #se a energia for nula, utilizamos apenas um array de zeros
            
        fft_mag_pos = fft_mag_norm[fft_mag_norm >0] #filtramos pelos valores da magnitude positivos
        if len(fft_mag_pos) >0:
            entropia= -np.sum(fft_mag_pos * np.log2(fft_mag_pos)) #medimos a incerteza do espectro.
            features.append(entropia)
        else:
            features.append(0)
    return np.array(features) #devolvemos a lista convertida em array

def segmentation(data, janela_size=5, overlap= 0.5, fs=51.2):
    #Esta função serve como auxílio à segmentação das feautures
    print("A segmentar os dados...\n")
    
    tam_janela= int(janela_size*fs) #calculamos o tamanho da janela em amostras
    passo= int(tam_janela * (1-overlap)) #vamos ver quanto é que a janela vai avançar
    
    segmentos= []
    labels= []
    dispositivos= []
    data_sorted= data[data[:,10].argsort()] #ordenar os dados com base na timestamp para estarem de ordem correta
    for disp_id in np.unique(data_sorted[:,0]): #percorrer cada dispositivo
        disp_data= data_sorted[data_sorted[:,0]==disp_id] #filtrar os dados para obtermos apenas as linhas que pertencem ao dispositivo atual
        i=0
        while i+tam_janela <=len(disp_data): #enquanto a janela não ultrapassar o tamanho dos dados
            segmento= disp_data[i:i+tam_janela] #criamos o segmento, extraindo as linhas de i até i+ tam-janela
            atividades_seg = np.unique(segmento[:,11])#encontrar os unique labels
            
            if len(atividades_seg)==1:
                segmentos.append(segmento)
                labels.append(atividades_seg[0])
                dispositivos.append(disp_id)
            i += passo #vamos deslizar a janela
    print(f"Segmentação concluída: {len(segmentos)} segmentos criados.\n")
    return segmentos, np.array(labels), np.array(dispositivos)
        
def extrair_features(data):
    #Esta função concatena as features temporais e espectrais para obtermos uma matriz final
    segmentos,labels,dispositivos= segmentation(data)

    if len(segmentos)==0:
        print("Não foi encontrado nenhum segmento!\n")
        return None, None, None
    
    print("A extrair features de cada segmento...\n")
    features_matrix= []
    for i, seg in enumerate(segmentos): #iteramos cada segmento criado 
        feat_temp= features_temporais(seg)
        feat_spec= features_espectrais(seg)
        features_complete= np. concatenate((feat_temp, feat_spec))
        features_matrix.append(features_complete)
        
    return np.array(features_matrix), np.array(labels), np.array(dispositivos) #convertemos a features_matrix num array
    #cada linha é um segmento e cada coluna uma feature

#As próximas funções servem para a implementação do PCA e para exemplificar como é que vão ser extraídas as features relativas à compressão.
#Sinoninamente, correspondem aos exercícios 4.3 e 4.4
def apply_pca(features_matrix, n_components=None):
    scaler= StandardScaler()
    features_normalizadas= scaler.fit_transform(features_matrix) #feature_set pronto para ser utilizado no PCA
    #as duas linhas de código acima servem para preparar os dados antes de se aplicar o PCA, vamos normalizar
    
    pca = PCA(n_components=n_components) #criamos uma instância de PCA
    features_pca= pca.fit_transform(features_normalizadas)#aplicamos o PCA aos dados normalizados
    
    var_acumulada= 0
    pc_75=None #variável que vai guardar o número de componentes necessárias para atingir 75% de variância explicada

    for i,var in enumerate(pca.explained_variance_ratio_): #vai percorrer um array  que contém a paercentagem de variância de cada componente
        var_acumulada +=var #vai adicionando a variância de cada componente ao total acumulado
        
        if i < 15 or (i + 1) % 10 == 0: 
            print(f"PC{i+1:<10} {var*100:<15.2f}{var_acumulada*100:<15.2f}") #print para mostrar a importância de cada componente
            
        if var_acumulada >= 0.75 and pc_75 is None: #verificamos se não atingiu ainda os 75%
            pc_75= i+1 #guarda a quantidade de componentes necessárias para atingir os 75%
            print("-" * 55)
            print(f"PC{i+1:<10} {var*100:<15.2f}{var_acumulada*100:<15.2f} <-- Limiar de 75% atingido")
            print("-" * 55)
            
    
    print(f"\nPara explicarmos 75% do feature set, devemos utilizar {pc_75} componentes principais.\n")
        
            
    return pca, features_pca, scaler, pc_75

#Esta função serve para exemplificar como é que o PCA atua sobre um intervalo específico  
def example_pca(pca, features_matrix, scaler,idx_exemplo=0,n_components_75=None): #vamos testar o primeiro segmento
    
    features_original= features_matrix[idx_exemplo]#selecionamos esse segmento da matriz original
    features_normalizadas= scaler.transform(features_original.reshape(1,-1)) #aplicamos o normalizador de zscore
    
    features_pca_full = pca.transform(features_normalizadas) #de seguida, aplicamos o PCA
    
    print(f"Features originais (segmento #{idx_exemplo}, 10 primeiras):")
    print(f"  {features_original[:10]}")
    
    print(f"\nFeatures normalizadas (segmento #{idx_exemplo}, 10 primeiras):")
    print(f"  {features_normalizadas[0, :10]}")
    
    print(f"\nFeatures PCA (Full) (segmento #{idx_exemplo}, 10 primeiras):")
    print(f"  {features_pca_full[0, :10]}")
    
    if n_components_75:
        features_pca_75= features_pca_full[0, :n_components_75] #cortamos o vetor features_pca_full para obter apenas as primeiras n_components_75  
        print(f"\nFeatures PCA (reduzidas para 75% variação) para o exemplo {idx_exemplo}:")
        print(f"  Dimensões: {len(features_pca_75)}")
        print(f"  Valores (primeiras 10): {features_pca_75[:10]}\n")
        

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
# --- MODIFICADA: Removida a variável global, agora recebe parâmetros ---
def print_selection_analysis(n_features_original, n_componentes_pca):
    """
    Imprime a análise de vantagens e limitações da Seleção de Features (Req 4.6.2)
    """
    print(f"\n{'='*80}")
    print(f"REQUISITO 4.6.2: VANTAGENS E LIMITAÇÕES DA SELEÇÃO DE FEATURES")
    print(f"{'='*80}")
    
    print("\n💡 VANTAGENS:")
    print("  • Interpretabilidade: Esta é a maior vantagem sobre o PCA. O modelo final "
          "usa as features originais (ex: 'média do giroscópio-x', 'std da aceleração-z'). "
          "Podemos interpretar *quais* características físicas são importantes.")
    print("  • Eficiência: Criar um modelo com 10 features é muito mais rápido do que "
          f"com {n_features_original} (ou mesmo com as "
          f"{n_componentes_pca} do PCA).")
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