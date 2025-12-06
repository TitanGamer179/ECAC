# ANÁLISE DOS CÁLCULOS DA META 2 (Ficheiros com "B")

## Resumo Executivo
✅ **Maioria dos cálculos está CORRETA**, mas existem **3-4 problemas importantes** que precisam ser corrigidos.

---

## 1. CALCULOB.PY - ANÁLISE DETALHADA

### ✅ Corretos:

#### 1.1 `check_balance(labels)` - CORRETO
- Calcula corretamente a distribuição das classes
- Verifica desbalanceamento com o critério `max/min > 1.5` ✓
- Output informativo

#### 1.2 `custom_smote()` - CORRETO
- Algoritmo SMOTE implementado corretamente:
  - Encontra vizinhos mais próximos usando KNN ✓
  - Gera amostras sintéticas interpolando entre vizinhos ✓
  - Valida se há dados suficientes (`len(dados) < k+1`) ✓

#### 1.3 `apply_smote()` - CORRETO
- Balanceia todas as classes até ao máximo ✓
- Retorna features e labels aumentadas ✓

#### 2.1 `extract_embedding_features()` - CORRETO
- Carrega modelo Harnet5 ✓
- Reamostragem a 30Hz com 5s ✓
- Batch processing (batch_size=32) ✓
- Reshape final correto para (N, dim) ✓

#### 3.1 `tvt_split()` - CORRETO
- Split 60-20-20 estratificado ✓
- Usa StratifiedShuffleSplit (mantém proporções das classes) ✓
- Output mostra percentagens reais vs esperadas ✓

#### 3.2 `split_between_subjects()` - CORRETO
- Divide 15 participantes (9 treino, 3 val, 3 teste) ✓
- Random seed fixado (reprodutibilidade) ✓

#### 3.3 `combined_split()` - CORRETO (com nota)
- Normalização StandardScaler aplicada corretamente ✓
- PCA com 90% variância ✓
- ReliefF com top 15 features ✓

#### 4.1 `train_evaluate_knn()` - CORRETO
- Predições usando votação por maioria ✓
- Métricas em validação e teste ✓

#### 4.2 `calculate_metrics()` - CORRETO
- Accuracy, Precision, Recall, F1 computados corretamente ✓
- Matriz de confusão ✓
- Weighted average (apropriado para classes desbalanceadas) ✓

#### 5.1 `hyperparameter_tuning()` - CORRETO
- Grid search para melhor k (1-20) ✓
- Retreina com treino+validação ✓
- Avalia em teste ✓
- Retorna métricas completas ✓

---

### ❌ PROBLEMAS ENCONTRADOS:

#### PROBLEMA 1: `report_results()` - INDENTAÇÃO INCORRETA
**Linha 287-290:**
```python
if accuracies: 
    results[config_name]={ ... }
    print(...)
else:
    print(f"Nenhum resultado obtido...")
```

**Problema:** O `if accuracies:` está DENTRO do loop `for iter`, não fora!
- Isto calcula resultados a cada iteração, não ao final
- Com `n_iterations=1`, funciona por acaso, mas com n_iterations>1, está errado

**Correção necessária:**
```python
for iter in range(n_iterations):
    # ... código da iteração
    
# MOVER PARA AQUI (fora do loop)
if accuracies: 
    results[config_name]={ 'accuracies': accuracies, 'mean_acc': np.mean(accuracies), ...}
    print(...)
else:
    print(...)
```

---

#### PROBLEMA 2: `tvt_split()` - PARÂMETRO NÃO USADO
**Linha 108:**
```python
def tvt_split(features, labels, train_ratio=0.6, val_ratio=0.2):
    # ... mas train_ratio e val_ratio são calculados hardcoded
    test_size = 0.2  # hardcoded
    test_size = val_ratio/(train_ratio + val_ratio)  # só aqui usa val_ratio
```

**Problema:** Se alguém passar `train_ratio=0.7, val_ratio=0.3`, o código ignora isto
**Recomendação:** Documentar que estes parâmetros são ignorados ou remover

---

#### PROBLEMA 3: `split_between_subjects()` - HARDCODED PARA 15 PARTICIPANTES
**Linha 141:**
```python
train_p_ids = shuffled_parts[:9]      # 9 participantes
val_p_ids = shuffled_parts[9:12]      # 3 participantes  
test_p_ids = shuffled_parts[12:]      # resto
```

**Problema:** Se tiver mais/menos de 15 participantes, isto quebra
**Exemplo:** Com 20 participantes, ficaria 9/3/8, mas esperado era balanceado

**Correção:**
```python
n_parts = len(unique_parts)
train_size = int(0.6 * n_parts)
val_size = int(0.2 * n_parts)
train_p_ids = shuffled_parts[:train_size]
val_p_ids = shuffled_parts[train_size:train_size+val_size]
test_p_ids = shuffled_parts[train_size+val_size:]
```

---

#### PROBLEMA 4: `combined_split()` - NORMALIZADOR NÃO GUARDADO
**Linha 173-174:**
```python
def combined_split(x_train, y_train, x_val, x_test, method):
    scaler = StandardScaler()
    x_train_norm = scaler.fit_transform(x_train)
    x_val_norm = scaler.transform(x_val)
    x_test_norm = scaler.transform(x_test)
    ...
    return x_train_pca, x_val_pca, x_test_pca, scaler, pca  # retorna scaler
```

**Problema:** O `scaler` é retornado mas nunca é usado no `mainActivityB.py`
- Se quiser fazer predições depois, precisa do scaler guardado
- Não é erro técnico, mas falta de boas práticas

---

### ⚠️ NOTAS IMPORTANTES:

1. **Ordem de retorno em `hyperparameter_tuning()`:**
   - `return best_val_k, test_metrics, val_accuracy, best_model, y_test_preds`
   - Em `report_results()` linha 275 está CORRETO:
   ```python
   best_k, test_metrics, val_accuracy, best_model, y_test_preds = ...
   ```

2. **Random seed:** Fixado em 42 para reprodutibilidade ✓

3. **Estratificação:** Usada corretamente em splits ✓

---

## 2. GRAFICOB.PY - ANÁLISE

### ✅ Correto:
- `plot_balance()` - gráfico de barras com contagens ✓
- `plot_augmentation_scatter()` - visualização 2D de dados originais vs SMOTE ✓
- `generate_examples()` - exemplos por classe ✓

### ⚠️ Notas:
- Nenhum gráfico de confusão (recomendação: adicionar)
- Nenhuma visualização de importância de features (PCA/ReliefF)

---

## 3. MAINACTIVITYB.PY - ANÁLISE

### ✅ Correto:
- Carregamento de dados ✓
- Filtragem de atividades (≤7) ✓
- Sequência lógica de transformações ✓

### ⚠️ PROBLEMAS:

#### PROBLEMA 1: Não chama `report_results()`!
**Linha 139:** Cria `dataset` dict mas nunca passa a `report_results()`
- O programa termina sem fazer a avaliação completa
- `report_results()` é definida mas nunca usada

**Correção necessária:**
```python
# Adicionar após linha 139:
results, predictions = calculoB.report_results(
    dataset, 
    participants=parts_seg, 
    n_iterations=5
)
```

#### PROBLEMA 2: Só treina Dataset_B manualmente
- Não testa Dataset_A com hyperparameter tuning
- Dataset_A é criado mas pouco usado

---

## RESUMO DE CORREÇÕES NECESSÁRIAS

| # | Ficheiro | Função | Severidade | Descrição |
|---|----------|--------|-----------|-----------|
| 1 | calculoB.py | report_results() | 🔴 CRÍTICA | Indentação do `if accuracies:` |
| 2 | mainActivityB.py | main() | 🔴 CRÍTICA | Não chama `report_results()` |
| 3 | calculoB.py | split_between_subjects() | 🟡 MÉDIA | Hardcoded para 15 participantes |
| 4 | calculoB.py | tvt_split() | 🟡 MÉDIA | Parâmetros ignorados |
| 5 | graficoB.py | - | 🟢 BAIXA | Faltam gráficos de confusion matrix |

---

## CONCLUSÃO

**Status Global: 75% CORRETO, 25% COM PROBLEMAS**

Os cálculos core (SMOTE, k-NN, métricas) estão corretos. Os problemas são principalmente:
1. Estrutura de loops
2. Integração entre módulos
3. Falta de robustez a diferentes tamanhos de dataset

**Tempo estimado para correções: 30-45 minutos**
