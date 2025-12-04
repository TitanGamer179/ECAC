import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


#1.Data Augmentation
def generate_examples(labels, title="Balanceamento de Classes (Atividades 1-7)"):
    labels_int= labels.astype(int)
    count = Counter (labels_int)
    
    sorted_labels= sorted(count.keys())
    counts_sorted= [count[i] for i in sorted_labels]
    
    plt.figure(figsize=(10,6))
    bars= plt.bar([str(i) for i in sorted_labels], counts_sorted, color= 'pink')
    
    plt.title(title)
    plt.xlabel("ID da Atividades")
    plt.ylabel("Número de Amostras")
    
    for bar in bars:
        yval= bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 5, yval, ha='center', va='bottom')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()