import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

#1.1:Visualização do Balanceamento das Classes
def plot_balance(labels, title="Distribuição de Classes"):
    unique, counts = np.unique(labels, return_counts=True)
    plt.figure(figsize=(10, 6))
    plt.bar(unique, counts, color='skyblue', edgecolor='black')
    plt.xlabel('Classes (Atividades)')
    plt.ylabel('Número de Amostras')
    plt.title(title)
    plt.xticks(unique)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i, count in enumerate(counts):
        plt.text(unique[i], count + (max(counts)*0.01), str(count), ha='center')
    plt.show()

#1.3: Visualização de Exemplos Após Augmentação
def generate_examples(features, labels, num_examples=5):
    unique_classes = np.unique(labels)
    plt.figure(figsize=(15, len(unique_classes) * 3))
    for i, cls in enumerate(unique_classes):
        class_features = features[labels == cls]
        n_available = len(class_features)
        if n_available > 0:
            n_select = min(num_examples, n_available)
            selected_indices = np.random.choice(n_available, n_select, replace=False)    
            for j, idx in enumerate(selected_indices):
                plt.subplot(len(unique_classes), num_examples, i * num_examples + j + 1)
                plt.plot(class_features[idx])
                if j == 0:
                    plt.ylabel(f'Classe {cls}', fontsize=12)
                if i == 0:
                    plt.title(f'Exemplo {j+1}', fontsize=12)
    plt.tight_layout()
    plt.show()