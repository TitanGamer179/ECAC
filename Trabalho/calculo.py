import math
import numpy as np

def calcular_tratamento_outliers(data,j):
    t_categoria=[]
    for i in range(len(data)):
        t=math.sqrt(data[i][j]**2+data[i][j+1]**2+data[i][j+2]**2)
        t_categoria.append(t)
    return t_categoria

def calcular_all(data):
    all_t=[]
    for j in range(1, len(data[0]), 3):
        if j<len(data[0])-2:
            t=calcular_tratamento_outliers(data,j)
            all_t.append(t)
    return all_t

def densidade_outliers(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iiqr=q3 - q1
    lim_inferior = q1 - 1.5 * iiqr
    lim_superior = q3 + 1.5 * iiqr
    return (data<lim_inferior)|(data>lim_superior)

def calcular_densidade_outliers(data,id):
    dados=data[:,0]==id
    dados_id=data[dados]
    atividades=sorted(np.unique(dados_id[:,11]))
    densidade=[]
    for a in atividades:
        i_ativ=dados_id[:,11]==a
        mod_acc=dados_id[i_ativ,12]
        mod_gyro=dados_id[i_ativ,13]
        mod_mag=dados_id[i_ativ,14]
        out_acc=densidade_outliers(mod_acc)
        out_gyro=densidade_outliers(mod_gyro)
        out_mag=densidade_outliers(mod_mag)
        n_total=len(mod_acc)
        n_outliers=np.sum(out_acc|out_gyro|out_mag)
        densidade.append((n_outliers/n_total)*100)
    return densidade

def outliers_zscore(data, threshold=3):
    mean = np.mean(data)
    std_dev = np.std(data)
    z_scores = (data - mean) / std_dev
    return np.abs(z_scores) > threshold

def pao: