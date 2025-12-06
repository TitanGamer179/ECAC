# Teste de Friedman + Teste T de Student Pareado
## Análise Estatística - Meta 2 (Alíneas 5.3 e 5.4)

---

## 📊 Visão Geral

A implementação agora utiliza **dois testes complementares** para análise estatística rigorosa:

1. **Teste de Friedman (5.3)** - Análise global
2. **Teste T de Student Pareado (5.4)** - Análise específica de pares

---

## 1️⃣ Teste de Friedman (Alínea 5.3)

### O que é?
Teste **não-paramétrico** que avalia se há diferenças significativas entre **múltiplas configurações** sem assumir distribuição normal.

### Quando usar?
- Comparar **mais de 2 grupos/configurações** simultaneamente
- Dados são **emparelhados** (mesmas iterações para todas configs)
- **Não assume** distribuição normal dos dados
- Perfeito para **visão geral/análise inicial**

### Hipóteses
- **H0**: Todas as configurações têm desempenho similar
- **H1**: Pelo menos uma configuração difere significativamente

### Saída
```
Teste de Friedman:
  Estatística: 12.345
  p-value: 0.0234
  
  ✓ RESULTADO: Diferenças SIGNIFICATIVAS (p < 0.05)
    → Há diferenças entre configurações
    → Usar teste de Student pareado para pares específicos
```

### Interpretação
- **p < 0.05**: Rejeitar H0 - Há diferenças significativas ✓
- **p ≥ 0.05**: Não rejeitar H0 - Sem diferenças significativas ✗

---

## 2️⃣ Teste T de Student Pareado (Alínea 5.4)

### O que é?
Teste **paramétrico** que compara **pares específicos** de configurações para identificar exatamente quais diferem.

### Quando usar?
- Após Friedman significativo para **detalhar quais pares diferem**
- Comparar **2 configurações** por vez
- Dados são **pareados** (mesmas amostras em ambas)
- Assume distribuição **normal das diferenças**

### Hipóteses
- **H0**: Não há diferença entre o par
- **H1**: Há diferença entre o par

### Tipos de Comparações Implementadas

#### 1. **DATASETS** (Dataset_A vs Dataset_B)
```
Mesmo método + mesma estratégia
Exemplo: Dataset_A_all_Within-Subject vs Dataset_B_all_Within-Subject
Total: 3 métodos × 2 estratégias = 6 comparações
```

#### 2. **MÉTODOS** (all vs pca vs relief)
```
Mesmo dataset + mesma estratégia
Exemplo: Dataset_A_all_Within-Subject vs Dataset_A_pca_Within-Subject
Total: 2 datasets × 2 estratégias × C(3,2) = 12 comparações
```

#### 3. **ESTRATÉGIAS** (Within-Subject vs Between-Subject)
```
Mesmo dataset + mesmo método
Exemplo: Dataset_A_all_Within-Subject vs Dataset_A_all_Between-Subject
Total: 2 datasets × 3 métodos = 6 comparações
```

**Total de comparações pareadas: 24 testes**

### Saída Exemplo
```
Dataset_A_all_Within-Subject vs Dataset_B_all_Within-Subject
  t-statistic:    2.3456
  p-value:        0.0125 ✓ SIG
  Diferença:      0.0845

Dataset_A_pca_Within-Subject vs Dataset_A_all_Within-Subject
  t-statistic:   -0.5678
  p-value:        0.5823 ✗ NS
  Diferença:     -0.0245
```

### Interpretação
- **p < 0.05**: Rejeitar H0 - Pares diferem significativamente ✓ SIG
- **p ≥ 0.05**: Não rejeitar H0 - Pares não diferem ✗ NS

- **t > 0**: Primeira config melhor
- **t < 0**: Segunda config melhor
- **|t| maior**: Diferença mais significativa

---

## 🔄 Fluxo de Análise

```
1. FRIEDMAN (5.3)
   ├─ Há diferenças significativas globalmente?
   │  ├─ SIM (p < 0.05)
   │  │  └─→ Prosseguir para Student pareado
   │  └─ NÃO (p ≥ 0.05)
   │     └─→ Configs similares, mas ainda testar pares
   │
2. STUDENT PAREADO (5.4)
   ├─ Dataset_A vs Dataset_B
   ├─ Método all vs pca vs relief
   ├─ Strategy Within vs Between
   └─ Identificar EXATAMENTE quais pares diferem
```

---

## 📈 Conjunto de Dados Esperado

Para correta execução:
- **Configurações**: 12 (2 datasets × 3 métodos × 2 estratégias)
- **Acurácias por config**: Lista de valores de múltiplas iterações
- **Iterações mínimas**: 1 (mas recomendado 5-10 para robustez)

---

## 🎯 Principais Mudanças Implementadas

### `calculoB.py` - `analyze_results_5_3()`
✅ Adicionado teste de Friedman  
✅ Rankings de Friedman por configuração  
✅ Interpretação clara de significância  
✅ Guia para próximo teste  
✅ Retorna `(friedman_stat, friedman_p)`

### `calculoB.py` - `paired_t_test_5_4()`
✅ Parâmetro `friedman_p` para contexto  
✅ Prefácio explicando complementariedade dos testes  
✅ 24 comparações pareadas sistemáticas  
✅ Resumo de significância  
✅ Retorna `all_pairs` com detalhes

### `mainActivityB.py`
✅ Captura `friedman_stat, friedman_p` de 5.3  
✅ Passa `friedman_p` para 5.4  
✅ Fluxo integrado e transparente

---

## 💡 Exemplo de Interpretação Prática

### Cenário 1: Friedman Significativo
```
Teste de Friedman: p = 0.0234 (SIG)
├─ Conclusão: Há diferenças entre configurações
├─ Teste Student revela:
│  ├─ Dataset_B melhor que Dataset_A (p < 0.05)
│  ├─ PCA melhor que all (p < 0.05)
│  └─ Within-Subject similar a Between-Subject (p ≥ 0.05)
└─ Recomendação: Use Dataset_B com PCA
```

### Cenário 2: Friedman Não Significativo
```
Teste de Friedman: p = 0.6234 (NS)
├─ Conclusão: Sem diferenças globais
├─ Teste Student revela:
│  └─ Nenhum par significativo (todos p ≥ 0.05)
└─ Recomendação: Qualquer config é aceitável
```

---

## ⚠️ Notas Importantes

1. **Independência**: Friedman é global; Student identifica pares
2. **Robustez**: Com n_iterations > 1, testes são mais robustos
3. **Múltiplas Comparações**: Com 24 testes, considerar correção de Bonferroni se necessário
4. **Alpha Inflation**: p < 0.05 sem correção; com Bonferroni: p < 0.05/24 ≈ 0.002
5. **Dados**: Ambos assumem dados pareados (mesmas iterações)

---

## 📋 Checklist de Validação

- ✅ Teste de Friedman implementado em 5.3
- ✅ Rankings de Friedman calculados
- ✅ Teste T pareado em 5.4 com contexto de 5.3
- ✅ 3 tipos de comparações (datasets, métodos, estratégias)
- ✅ 24 pares comparados
- ✅ Integração em mainActivityB.py
- ✅ Documentação completa
- ✅ Pronto para execução

---

## 🚀 Como Executar

```bash
python mainActivityB.py
```

Saída esperada:
```
5.3: ANÁLISE COMPARATIVA DOS RESULTADOS
  1. Efeito do dataset
  2. Efeito do método
  3. Efeito da estratégia
  4. Ranking
  5. Robustez
  6. TESTE DE FRIEDMAN ← NEW!
  7. Conclusões

5.4: TESTE T DE STUDENT PAREADO
  📊 CONTEXTO DO TESTE DE FRIEDMAN (5.3) ← NEW!
  1. COMPARAÇÕES ENTRE DATASETS
  2. COMPARAÇÕES ENTRE MÉTODOS
  3. COMPARAÇÕES ENTRE ESTRATÉGIAS
  4. RESUMO ESTATÍSTICO
  5. INTERPRETAÇÃO
```

---

**Status**: ✅ Implementação completa e pronta para uso
