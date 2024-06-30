# Título: Reglas de reparto para el problema de la bancarrota
# Autor: Víctor Herrezuelo Paredes
# Fecha: Junio 2024
# Trabajo Fin de Grado - Grado en Administraciónn y Dirección de Empresas
# Universidad de Valladolid
# Descripción: este script calcula el reparto de un presupuesto E entre varios agentes en función de sus reclamaciones y siguiendo cinco reglas de reparto: proporcional, igual ganancia, igual pérdida, talmud y orden de llegada. Además, muestra una tabla con los resultados y gráficas con el porcentaje de reclamaciones obtenidas por cada agente en función del porcentaje del presupuesto sobre la reclamación agregada (E/C).


import pandas as pd
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk
import numpy as np
import itertools

# Funciones
def leer_datos():
    # Función para leer un fichero CSV con las reclamaciones de los agentes
    root = Tk() 
    root.withdraw()
    filepath = filedialog.askopenfilename(title="Selecciona un fichero CSV con reclamaciones")
    return pd.read_csv(filepath, sep=';')

def calcular_lambda(reclamaciones, E, es_talmud=False):
    # Función para calcular el valor de lambda
    # Para poder resolver la ecuación, se utiliza una búsqueda binaria
    if es_talmud:
        # En el caso de la regla de Talmud, las reclamaciones se dividen entre 2
        reclamaciones = reclamaciones / 2

    low, high = 0, max(reclamaciones) # Inicializamos los valores de low y high
    epsilon = 1e-6  # Tolerancia para la solución

    while high - low > epsilon:
        mid = (low + high) / 2
        suma_minimos = np.sum(np.minimum(mid, reclamaciones))
        if suma_minimos < E:
            low = mid
        else:
            high = mid

    return (low + high) / 2

def calcular_mu(reclamaciones, E, es_talmud=False):
    # Función para calcular el valor de mu
    # Para poder resolver la ecuación, se utiliza una búsqueda binaria
    low, high = 0, max(reclamaciones)
    epsilon = 1e-6  # Tolerancia para la solución

    while high - low > epsilon:
        mid = (low + high) / 2
        if es_talmud:
            suma_maximos = np.sum(np.maximum(reclamaciones / 2, reclamaciones - mid))
        else:
            suma_maximos = np.sum(np.maximum(0, reclamaciones - mid))
        
        if suma_maximos > E:
            low = mid
        else:
            high = mid
                        
    return (low + high) / 2

def calcular_reparto_proporcional(reclamaciones, E):
    # Función para calcular el reparto proporcional
    C = reclamaciones.sum()
    return (E / C) * reclamaciones

def calcular_reparto_igual_ganancia(reclamaciones, E):
    # Función para calcular el reparto con igual ganancia
    lambda_value = calcular_lambda(reclamaciones, E)
    return np.minimum(reclamaciones, lambda_value)

def calcular_reparto_igual_perdida(reclamaciones, E):
    #  Función para calcular el reparto con igual pérdida
    mu_value = calcular_mu(reclamaciones, E)
    return np.maximum(reclamaciones - mu_value, 0)

def calcular_reparto_talmud(reclamaciones, E):
    # Función para calcular el reparto con la regla de Talmud
    C = reclamaciones.sum()
    half_c = C / 2
    if E <= half_c:
        lambda_value = calcular_lambda(reclamaciones, E, es_talmud=True)
        return np.minimum(reclamaciones / 2, lambda_value)
    else:
        mu = calcular_mu(reclamaciones, E, es_talmud=True)
        return np.maximum(reclamaciones / 2, reclamaciones - mu)

def calcular_reparto_orden_llegada(reclamaciones, E):
    # Función para calcular el reparto según la regla del orden de llegada
    n = len(reclamaciones)
    permutaciones = list(itertools.permutations(range(n))) # Todas las permutaciones posibles
    distribucion = np.zeros(n)
    for perm in permutaciones:
        # Calculamos el reparto para cada permutación
        E_temp = E
        temp_distribucion = np.zeros(n)
        for agente in perm:
            # Asignamos el mínimo entre la reclamación del agente y el presupuesto restante
            asignado = min(reclamaciones[agente], E_temp)
            temp_distribucion[agente] = asignado
            E_temp -= asignado
            if E_temp <= 0:
                break
        distribucion += temp_distribucion
    return distribucion / len(permutaciones) # Devolvemos la media de los repartos

def calcular_reparto(reclamaciones, E, regla):
    # Función para calcular el reparto según la regla especificada
    if regla == 'Proporcional':
        return calcular_reparto_proporcional(reclamaciones, E)
    elif regla == 'Igual Ganancia':
        return calcular_reparto_igual_ganancia(reclamaciones, E)
    elif regla == 'Igual Pérdida':
        return calcular_reparto_igual_perdida(reclamaciones, E)
    elif regla == 'Talmud':
        return calcular_reparto_talmud(reclamaciones, E)
    elif regla == 'Orden de Llegada':
        return calcular_reparto_orden_llegada(reclamaciones, E)

def dibujar_graficas(resultados, reclamaciones, C, E):
    # Función para dibujar las gráficas con los resultados
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    reglas = ['Igual Ganancia', 'Igual Pérdida', 'Talmud', 'Orden de Llegada']
    porcentajes = np.linspace(0, 1, 11)  # Coordenadas x (0, 0.1, ..., 1)
    E_porcentaje = (E / C) * 100

    # Dibujamos las gráficas
    for i, regla in enumerate(reglas):
        ax = axs[i // 2, i % 2]
        for j, agente in enumerate(resultados['Agente']):
            porcentaje_obtenido = [0.0]  # Comenzamos con (0,0)
            for p in porcentajes[1:-1]:  # Excluimos el primer y último punto
                E_temp = p * C
                repartido = calcular_reparto(reclamaciones, E_temp, regla)
                porcentaje_obtenido.append((repartido[j] / reclamaciones[j]) * 100 if reclamaciones[j] != 0 else 0)
            porcentaje_obtenido.append(100.0)  # Terminamos con (1,1)
            
            ax.plot(porcentajes * 100, porcentaje_obtenido, label=f'{agente} ({reclamaciones.iloc[j]})', marker='o')

        ax.axvline(x=E_porcentaje, color='r', linestyle='--', label=f'Presupuesto {E}')
        ax.plot([0, 100], [0, 100], 'k--', label='Reparto proporcional')
        ax.set_title(f'Reparto según la regla de {regla}')
        ax.set_xlabel('% del presupuesto sobre la reclamación agregada (E/C)')
        ax.set_ylabel('% de las reclamaciones obtenidas por cada ')
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 100])
        ax.xaxis.set_major_locator(plt.MultipleLocator(10))
        ax.yaxis.set_major_locator(plt.MultipleLocator(10))
        ax.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)
        ax.legend(title="Agente (Reclamación)")

    plt.tight_layout()
    plt.show()

def imprimir_tabla(resultados, reclamaciones, E):
    # Función para imprimir la tabla con los resultados
    reglas = ['Proporcional', 'Igual Ganancia', 'Igual Pérdida', 'Talmud', 'Orden de Llegada']
    tabla = pd.DataFrame({'Agente': resultados['Agente'], 'Reclamación': reclamaciones})

    for regla in reglas:
        repartos = calcular_reparto(reclamaciones, E, regla)
        porcentajes = (repartos / reclamaciones) * 100
        tabla[f'{regla}'] = [f'{reparto:.2f} ({porcentaje:.2f}%)' for reparto, porcentaje in zip(repartos, porcentajes)]
    
    print(tabla)

# Main
def main():
    datos = leer_datos() # Leemos los datos de las reclamaciones
    reclamaciones = datos['reclamacion']
    C = reclamaciones.sum()
    E = float(input("Introduce el presupuesto E: "))
    print(f"La suma total de todas las reclamaciones (C) es: {C}")

    # Guardamos los datos en un DataFrame
    resultados = pd.DataFrame({
        'Agente': datos['agente_nombre']
    })

    imprimir_tabla(resultados, reclamaciones, E) # Calculamos y mostramos los resultados
    dibujar_graficas(resultados, reclamaciones, C, E) # Dibujamos las gráficas

if __name__ == '__main__':
    main()
