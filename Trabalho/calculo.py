import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from scipy.stats import kstest, f_oneway, kruskal
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis

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
        
    
            
           
