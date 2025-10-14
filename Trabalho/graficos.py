import matplotlib.pyplot as plt

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