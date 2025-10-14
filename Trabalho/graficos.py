import matplotlib.pyplot as plt

# Função para criar boxplots das atividades
def bloxplot_activity(data,t):
    text=['acceleração','giroscópio','magnetômetro']
    fig, axs=plt.subplots(1,len(t),figsize=(15,6))
    for i in range(len(t)):
        axs[i].boxplot(data, t[i])
        axs[i].set_title(f'Atividade por {text[i]}')
        axs[i].set_xlabel('Coluna 12 (data)')
        axs[i].set_ylabel(f'Time[{text[i]}]')
        axs[i].grid(True)
    plt.tight_layout()
    plt.show()
    
# Função para criar gráficos de dispersão dos outliers detectados pelo Z-score
def scatter_outliers_zscore(data, outliers):
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(data)), data, label='Data Points', color='blue')
    plt.scatter([i for i in range(len(data)) if outliers[i]], 
                [data[i] for i in range(len(data)) if outliers[i]], 
                color='red', label='Outliers', marker='x')
    plt.title('Outlier Detection using Z-Score')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()