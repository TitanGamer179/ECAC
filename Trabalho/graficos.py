import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Função para criar boxplots das atividades
def boxplot_activity(all_data):
    device_ids=sorted(np.unique(all_data[:,0]))
    magnitudes_info = {12: 'Aceleração',13: 'Giroscópio',14: 'Magnetómetro'}
    for dev_id in device_ids:
        fig,axs=plt.subplots(1,3, figsize=(20, 6), sharey=False)
        fig.suptitle(f'Boxplots for Device {int(dev_id)}', fontsize=16)
        device_data=all_data[all_data[:,0]==dev_id]
        activity_labels=sorted(np.unique(device_data[:,11]))
        for idx, (col, title) in enumerate(magnitudes_info.items()):
            data_to_plot=[device_data[device_data[:,11]==a,col] for a in activity_labels]
            axs[idx].boxplot(data_to_plot)
            axs[idx].set_title(f'Módulo de {title}')
            axs[idx].set_xlabel('Atividade')
            axs[idx].set_ylabel('Módulo')
            axs[idx].grid(True)
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

# Função para criar gráficos de dispersão dos outliers detectados pelo Z-score
def scatter_outliers_zscore(device_data, outliers,k,magnitude_col,magnitude_name):
    plt.figure(figsize=(12, 7))
    inlier_data = device_data[~outliers]
    outlier_data = device_data[outliers]
    plt.scatter(inlier_data[:, 11], inlier_data[:, magnitude_col], color='blue', label='Inliers', alpha=0.5, s=10)
    plt.scatter(outlier_data[:, 11], outlier_data[:, magnitude_col], color='red', label='Outliers', marker='x')
    plt.title(f'Deteção de Outliers com Z-Score (k={k}) - Módulo do {magnitude_name}')
    plt.xlabel('Atividade')
    plt.ylabel('Valor do Módulo')
    plt.xticks(sorted(np.unique(device_data[:, 11]).astype(int)))
    plt.legend()
    plt.grid(True)
    plt.show()

# Função para criar boxplots para todos os dispositivos
def boxplot_all_devices(all_data):
    magnitudes_info = {12: 'Aceleração', 13: 'Giroscópio', 14: 'Magnetómetro'}
    activity_labels = sorted(np.unique(all_data[:, 11]))
    fig, axs = plt.subplots(1, 3, figsize=(20, 6), sharey=False)
    fig.suptitle('Boxplots para Todos os Dispositivos', fontsize=16)
    for idx, (col, title) in enumerate(magnitudes_info.items()):
        data_to_plot = [all_data[all_data[:, 11] == a, col] for a in activity_labels]
        axs[idx].boxplot(data_to_plot)
        axs[idx].set_title(f'Módulo de {title}')
        axs[idx].set_xlabel('Atividade')
        axs[idx].set_ylabel('Módulo')
        axs[idx].grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
def visualizar_outliers_3d(data_3d, outliers_mask, title):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    inliers = data_3d[~outliers_mask]
    outliers = data_3d[outliers_mask]
    ax.scatter(inliers[:, 0], inliers[:, 1], inliers[:, 2],c='blue', label='Inliers', alpha=0.4, s=15)
    ax.scatter(outliers[:, 0], outliers[:, 1], outliers[:, 2],c='red', label='Outliers', marker='x', s=50)
    ax.set_title(title)
    ax.set_xlabel('Acelerómetro Eixo X')
    ax.set_ylabel('Acelerómetro Eixo Y')
    ax.set_zlabel('Acelerómetro Eixo Z')
    ax.legend()
    plt.show()
    
def visualizar_clusters_kmeans_3d(data_3d, labels, n_clusters, title):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2],
                         c=labels, cmap='viridis', s=15, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel('Acelerómetro Eixo X'); ax.set_ylabel('Acelerómetro Eixo Y')
    ax.set_zlabel('Acelerómetro Eixo Z')
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    plt.show()

#Vamos criar um boxplot (para verificar os testes estatísticos) e histogramas(para ajudar com a vizualização do ktest)
def plot_testes_significativos_mpl(data):
    print("A gerar gráficos de justificação (Matplotlib) para a Alínea 4.1...")
    variaveis = {
        12: 'Módulo Aceleração',
        13: 'Módulo Giroscópio',
        14: 'Módulo Magnetómetro'
    }
    
    #Histograma
    dados_para_hist = []
    labels_para_hist = []
            
    for ativ in atividades:
        dados_ativ = disp_dados[disp_dados[:, 11] == ativ, var_idx]
        # Só plotamos se tivermos dados suficientes 
        if len(dados_ativ) > 2:
            dados_para_hist.append(dados_ativ)
            labels_para_hist.append(int(ativ))
            
        if not dados_para_hist:
            print(f"Sem dados para plotar histogramas para {var_nome}")
            continue
                
        # Calcular o tamanho da grelha 
        n_plots = len(labels_para_hist)
        n_cols = 4
        n_rows = int(np.ceil(n_plots / n_cols))
            
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes = axes.flatten() 
            
        for i in range(n_plots):
            ax = axes[i] # O 'eixo' (área de plotagem) atual
            dados_ativ = dados_para_hist[i]
            label_ativ = labels_para_hist[i]
                
            ax.hist(dados_ativ, bins=30, alpha=0.7, density=True, label='Dados')
                
            #Tentar sobrepor a curva normal ideal (o que o kstest compara)
            if np.std(dados_ativ) > 0:
                try:
                    mu, std = stats.norm.fit(dados_ativ) # Encontra a média e std
                    xmin, xmax = ax.get_xlim()
                    x = np.linspace(xmin, xmax, 100)
                    p = stats.norm.pdf(x, mu, std) # Cria a curva normal
                    ax.plot(x, p, 'r--', linewidth=2, label='Curva Normal')
                except Exception:
                    pass # Ignora se o 'fit' falhar

            ax.set_title(f'Atividade {label_ativ}')
            ax.set_xlabel(var_nome)
            ax.set_ylabel('Densidade')
                
            # Ocultar subplots que não foram usados
            for j in range(n_plots, len(axes)):
                axes[j].set_visible(False)
            
            fig.suptitle(f'Verificação de Normalidade para {var_nome}\nDispositivo {int(disp)}', 
                           y=1.03, fontsize=16)
        plt.tight_layout()
        plt.show()
        
    #Boxplots
    dispositivos = sorted(np.unique(data[:, 0]))
    atividades = sorted(np.unique(data[:, 11])) # Lista de todas as IDs de atividades
    for disp in dispositivos:
        print(f"\n--- A gerar gráficos para o Dispositivo {int(disp)} ---")
        
        # Filtrar dados para o dispositivo atual
        disp_dados = data[data[:, 0] == disp]
        
        for var_idx, var_nome in variaveis.items():
            
            print(f"A processar variável: {var_nome}")
            
            dados_para_boxplot = []
            labels_para_boxplot = []
            
            for ativ in atividades:
                # Extrai os dados para esta atividade e variável
                dados_ativ = disp_dados[disp_dados[:, 11] == ativ, var_idx]
                if len(dados_ativ) > 0:
                    dados_para_boxplot.append(dados_ativ)
                    labels_para_boxplot.append(int(ativ)) # Guarda o ID da atividade
            
            if not dados_para_boxplot:
                print(f"Sem dados para plotar boxplot para {var_nome}")
                continue

            plt.figure(figsize=(16, 7))
            # Criar o boxplot
            plt.boxplot(dados_para_boxplot)
            
            # Definir os rótulos do eixo X para corresponderem às atividades
            plt.xticks(range(1, len(labels_para_boxplot) + 1), labels_para_boxplot)
            
            plt.title(f'Comparação de Atividades para {var_nome}\nDispositivo {int(disp)}', fontsize=16)
            plt.xlabel('ID da Atividade', fontsize=12)
            plt.ylabel(var_nome, fontsize=12)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.show()

            
#Gráfico que destaca a importância de cada vetor, mostrando exatamente onde é que +e atingido o 75%          
def plot_variancia_explicada_pca(pca, pc_75=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Variância explicada por componente
    n_components = len(pca.explained_variance_ratio_)
    ax1.bar(range(1, n_components+1), pca.explained_variance_ratio_ * 100)
    ax1.set_xlabel('Componente Principal')
    ax1.set_ylabel('Variância Explicada (%)')
    ax1.set_title('Variância Explicada por Componente Principal')
    ax1.grid(True, alpha=0.3)
    
    if pc_75:
        ax1.axvline(x=pc_75, color='r', linestyle='--', label=f'75% variância (PC{pc_75})')
        ax1.legend()
    
    # Variância acumulada
    var_acumulada = np.cumsum(pca.explained_variance_ratio_) * 100
    ax2.plot(range(1, n_components+1), var_acumulada, 'b-o', linewidth=2, markersize=4)
    ax2.axhline(y=75, color='r', linestyle='--', label='75%')
    ax2.set_xlabel('Número de Componentes')
    ax2.set_ylabel('Variância Acumulada (%)')
    ax2.set_title('Variância Acumulada')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    if pc_75:
        ax2.axvline(x=pc_75, color='r', linestyle='--', alpha=0.5)
        ax2.plot(pc_75, 75, 'ro', markersize=10)
    
    plt.tight_layout()
    plt.show()

#Gr´´afico que mostra o resultado da compressão aplicada aos segmentos
def plot_pca_2d(features_pca, labels, title="Visualização PCA 2D"):
    plt.figure(figsize=(12, 8))
    
    # Mapear labels para cores
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(features_pca[mask, 0], features_pca[mask, 1],
                   c=[colors[idx]], label=f'Atividade {int(label)}',
                   alpha=0.6, s=30)
    
    plt.xlabel('PC1 (Primeira Componente Principal)')
    plt.ylabel('PC2 (Segunda Componente Principal)')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_pca_3d(features_pca, labels, title="Visualização PCA 3D"):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Mapear labels para cores
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(features_pca[mask, 0], features_pca[mask, 1], features_pca[mask, 2],
                  c=[colors[idx]], label=f'Atividade {int(label)}',
                  alpha=0.6, s=30)
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

#Gráficos que plotam os scores das N melhores features
def plot_fisher_scores(fisher_scores, top_n=10):
    # Ordenar scores
    sorted_indices = np.argsort(fisher_scores)[::-1][:top_n]
    sorted_scores = fisher_scores[sorted_indices]
    
    plt.figure(figsize=(14, 8))
    bars = plt.bar(range(top_n), sorted_scores, color='steelblue', alpha=0.8)
    
    # Destacar top 10
    for i in range(min(10, top_n)):
        bars[i].set_color('darkred')
        bars[i].set_alpha(0.9)
    
    plt.xlabel('Ranking da Feature')
    plt.ylabel('Fisher Score')
    plt.title(f'Top {top_n} Features por Fisher Score')
    plt.xticks(range(top_n), [f'F{sorted_indices[i]}' for i in range(top_n)], rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
    
    print("\nTop 10 Features (Fisher Score):")
    for i in range(min(10, top_n)):
        print(f"{i+1}. Feature {sorted_indices[i]}: {sorted_scores[i]:.4f}")

def plot_relieff_weights(relieff_weights, top_n=10):

    # Ordenar weights
    sorted_indices = np.argsort(relieff_weights)[::-1][:top_n]
    sorted_weights = relieff_weights[sorted_indices]
    
    plt.figure(figsize=(14, 8))
    bars = plt.bar(range(top_n), sorted_weights, color='forestgreen', alpha=0.8)
    
    # Destacar top 10
    for i in range(min(10, top_n)):
        bars[i].set_color('darkgreen')
        bars[i].set_alpha(0.9)
    
    plt.xlabel('Ranking da Feature')
    plt.ylabel('ReliefF Weight')
    plt.title(f'Top {top_n} Features por ReliefF')
    plt.xticks(range(top_n), [f'F{sorted_indices[i]}' for i in range(top_n)], rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
    
    print("\nTop 10 Features (ReliefF):")
    for i in range(min(10, top_n)):
        print(f"{i+1}. Feature {sorted_indices[i]}: {sorted_weights[i]:.4f}")

#Gráfico que compara os resultados de maneira visual, utilizando um diagrama de Venn para mostrar as features mais comuns
def plot_comparacao_fisher_relieff(fisher_ranking, relieff_ranking, top_n=10):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Top N de cada método
    top_fisher = fisher_ranking[:top_n]
    top_relieff = relieff_ranking[:top_n]
    
    # Features selecionadas por Fisher
    ax1.barh(range(top_n), range(top_n, 0, -1), color='steelblue', alpha=0.7)
    ax1.set_yticks(range(top_n))
    ax1.set_yticklabels([f'F{top_fisher[i]}' for i in range(top_n)])
    ax1.set_xlabel('Ranking')
    ax1.set_title('Top 10 Features - Fisher Score')
    ax1.invert_xaxis()
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Features selecionadas por ReliefF
    ax2.barh(range(top_n), range(1, top_n+1), color='forestgreen', alpha=0.7)
    ax2.set_yticks(range(top_n))
    ax2.set_yticklabels([f'F{top_relieff[i]}' for i in range(top_n)])
    ax2.set_xlabel('Ranking')
    ax2.set_title('Top 10 Features - ReliefF')
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()
    
    # Diagrama de Venn 
    set_fisher = set(top_fisher)
    set_relieff = set(top_relieff)
    comum = set_fisher & set_relieff
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Círculo Fisher
    circle1 = plt.Circle((0.3, 0.5), 0.3, color='steelblue', alpha=0.3, label='Fisher')
    # Círculo ReliefF
    circle2 = plt.Circle((0.7, 0.5), 0.3, color='forestgreen', alpha=0.3, label='ReliefF')
    
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    
    # Textos
    ax.text(0.2, 0.5, f'{len(set_fisher - set_relieff)}', fontsize=20, ha='center')
    ax.text(0.5, 0.5, f'{len(comum)}', fontsize=20, ha='center', fontweight='bold')
    ax.text(0.8, 0.5, f'{len(set_relieff - set_fisher)}', fontsize=20, ha='center')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.legend(loc='upper center')
    ax.set_title(f'Features Comuns no Top {top_n}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n{len(comum)} features comuns: {sorted(comum)}")

