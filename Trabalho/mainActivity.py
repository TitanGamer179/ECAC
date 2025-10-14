import openfile
import calculo
import graficos

def main():
    all_data = openfile.open_all_files('C:\\Users\\User\\Python\\ECAC\\Trabalho\\diretoria')
    print("Arquivos abertos com sucesso!")
    resultados = calculo.calcular_all(all_data)
    print("Cálculos realizados com sucesso!")
    data12 = []
    for i in range(len(all_data)):
        data12.append(all_data[i][11])
    graficos.bloxplot_activity(data12, resultados)
    calculo_densidade = calculo.calcular_densidade_outliers(all_data)
    print("Densidade de outliers calculada com sucesso!")
    densidade=calculo.calcular_densidade_outliers(all_data,2)
    print("Densidade de outliers para a atividade 2:")
    for i, d in enumerate(densidade):
       print(f"Atividade {i}: {d:.2f}%")
    


if __name__ == "__main__":
    main()
