import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


#1.3: Visualização de Exemplos Após Augmentação
def generate_examples(features, labels, num_examples=5):
    unique_classes = np.unique(labels)
    plt.figure(figsize=(15, len(unique_classes) * 3))
    for i, cls in enumerate(unique_classes):
        class_features = features[labels == cls]
        selected_indices = np.random.choice(len(class_features), num_examples, replace=False)
        for j, idx in enumerate(selected_indices):
            plt.subplot(len(unique_classes), num_examples, i * num_examples + j + 1)
            plt.plot(class_features[idx])
            if j == 0:
                plt.ylabel(f'Classe {cls}', fontsize=12)
            if i == 0:
                plt.title(f'Exemplo {j+1}', fontsize=12)
    plt.tight_layout()
    plt.show()