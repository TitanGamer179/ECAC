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
def plot_augmentation_scatter(original_data, synthetic_data):
    plt.figure(figsize=(10, 7))
    plt.scatter(original_data[:, 0], original_data[:, 1], c='blue', label='Original', alpha=0.6, s=50)
    plt.scatter(synthetic_data[:, 0], synthetic_data[:, 1], c='red', marker='*', s=200, label='Sintético (SMOTE)')
    plt.title("Data Augmentation: Part 3, Ativ 4 (2 Primeiras Features)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

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

def plot_confusion_matrix(y_true, y_pred, title="Matriz de Confusão"):
    """Visualizar matriz de confusão."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=14)
    plt.colorbar()
    classes = sorted(np.unique(y_true))
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.ylabel('Verdadeira', fontsize=12)
    plt.xlabel('Predita', fontsize=12)
    
    # Adicionar valores na matriz
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.show()

def plot_feature_importance(importance, feature_names=None, top_n=15, title="Importância de Features"):
    """Visualizar importância das features (ReliefF ou PCA)."""
    if feature_names is None:
        feature_names = [f"F{i}" for i in range(len(importance))]
    
    sorted_idx = np.argsort(importance)[::-1][:top_n]
    plt.figure(figsize=(10, 6))
    plt.barh(range(top_n), importance[sorted_idx])
    plt.yticks(range(top_n), [feature_names[i] for i in sorted_idx])
    plt.xlabel('Importância')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()