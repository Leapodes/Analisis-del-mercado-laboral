import os
import sys
import warnings
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('TkAgg')

warnings.filterwarnings(
    "ignore", message=".*GeoJSON does not support open option DRIVER.*"
)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# Corrección para versiones de Pandas sin SettingWithCopyWarning en errors
try:
    from pandas.errors import SettingWithCopyWarning
    warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
except ImportError:
    warnings.filterwarnings("ignore", message=".*SettingWithCopyWarning.*")

# Codificación para leer simbolos y mayusculas
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Codigos INDEC: 32 = Ciudad Autónoma de Buenos Aires, 33 = Partidos del Gran Buenos Aires
AGLOMERADOS_COD = [32, 33]
NOMBRES_AGLOMERADOS = {32: "CABA", 33: "Conurbano Bonaerense"}

# 0. GENERADOR AUTOMÁTICO DE IPC TRIMESTRAL
def generar_ipc_trimestral():
    if not os.path.exists("Datos"):
        os.makedirs("Datos")

    archivo_ipc = "Datos/ipc_trimestral.csv"

    if not os.path.exists(archivo_ipc):
        print(
            "[+] Generando archivo 'Datos/ipc_trimestral.csv' a partir de datos del INDEC..."
        )
        variaciones_6x6 = [
            [1.3, 2.5, 2.4, 2.6, 1.3, 1.4],
            [1.7, 1.5, 2.0, 1.3, 1.2, 3.4],
            [1.6, 2.6, 2.5, 2.6, 1.9, 3.9],
            [2.8, 4.1, 6.6, 5.1, 2.9, 2.8],
            [2.8, 3.8, 4.8, 3.2, 3.0, 2.6],
            [2.1, 3.9, 5.8, 3.2, 4.1, 3.8],
            [1.9, 1.8, 3.6, 1.4, 1.5, 2.0],
            [1.6, 2.8, 2.8, 3.6, 3.0, 3.7],
            [3.3, 3.6, 5.2, 4.1, 3.4, 3.1],
            [3.1, 2.6, 3.8, 3.8, 2.3, 4.1],
            [3.9, 4.6, 6.7, 6.2, 4.8, 5.5],
            [7.4, 7.0, 6.0, 6.6, 5.0, 5.3],
            [6.0, 6.7, 7.8, 8.6, 8.0, 5.8],
            [6.2, 12.3, 12.2, 8.6, 12.9, 25.1],
            [19.6, 15.0, 11.5, 9.2, 4.3, 4.4],
            [4.0, 4.1, 3.7, 2.8, 2.6, 2.9],
            [2.0, 2.2, 3.9, 2.8, 1.5, 2.0],
            [1.9, 1.9, 2.1, 2.4, 2.5, 2.8],
            [2.8, 2.6, 3.4, 2.8, 2.3, 1.9],
            [2.3]
        ]

        # Para aplanarla a una sola lista simple dentro del script:
        variaciones_mensuales = [valor for fila in variaciones_6x6 for valor in fila]

        ipc_acumulado = []
        valor_actual = 100.0
        for var in variaciones_mensuales:
            valor_actual *= 1 + (var / 100.0)
            ipc_acumulado.append(valor_actual)
        registros = []
        idx = 0

        for anio in range(2020, 2026):
            for trimestre in range(1, 6):
                if anio == 2020 and trimestre == 0:
                    continue
                if idx >= len(ipc_acumulado):
                    break

                meses_trimestre = ipc_acumulado[idx : idx + 3]
                if len(meses_trimestre) > 0:
                    ipc_trimestral = sum(meses_trimestre) / len(meses_trimestre)
                    registros.append(
                        {
                            "ANO4": anio,
                            "TRIMESTRE": trimestre,
                            "IPC": round(ipc_trimestral, 2),
                        }
                    )
                idx += 3

        df_ipc = pd.DataFrame(registros)
        df_ipc.to_csv(archivo_ipc, index=False)
        print("Archivo IPC creado correctamente")


# 1. CARGA DE DATOS (2020-2026)

def cargar_datos():
    datos_dir = "Datos/"
    df_total = pd.DataFrame()

    for anio in range(19, 27):  # Carga de años 2020 (20) a 2026 (26)
        for trimestre in range(1, 6):
            if anio == 20 and trimestre == 0:
                continue
            elif anio == 26 and trimestre >= 3:
                continue  # Toma hasta T2 de 2026

            archivo = datos_dir + f"usu_individual_T{trimestre}{anio}.txt"

            try:
                df_datos = pd.read_csv(archivo, sep=";", encoding="latin1")
                df_total = pd.concat([df_total, df_datos], ignore_index=True)
                print(f"{trimestre}° Trimestre del año 20{anio} cargado")
            except Exception:
                pass

    return df_total


# 2. AJUSTE POR INFLACIÓN (P47T_real)

def ajustar_por_inflacion(df_total):
    try:
        ipc = pd.read_csv("Datos/ipc_trimestral.csv", encoding="utf-8")
    except Exception:
        print("No se encontró ipc_trimestral. Se utilizará ingreso nominal")
        df_total["P47T_real"] = pd.to_numeric(
            df_total["P47T"], errors="coerce"
        )
        return df_total

    ipc["ANO4"] = ipc["ANO4"].astype(int)
    ipc["TRIMESTRE"] = ipc["TRIMESTRE"].astype(int)

    base = ipc[(ipc["ANO4"] == 2024) & (ipc["TRIMESTRE"] == 4)]["IPC"].values
    base_val = base[0] if len(base) > 0 else ipc["IPC"].iloc[-1]

    df_total["ANO4"] = pd.to_numeric(df_total["ANO4"], errors="coerce")
    df_total["TRIMESTRE"] = pd.to_numeric(
        df_total["TRIMESTRE"], errors="coerce"
    )

    df_total = df_total.merge(ipc, on=["ANO4", "TRIMESTRE"], how="left")
    df_total["P47T"] = pd.to_numeric(df_total["P47T"], errors="coerce")
    df_total["P47T_real"] = df_total["P47T"] * (base_val / df_total["IPC"])

    return df_total


# 3. CÁLCULO DE TASAS LABORALES

def calcular_tasas(df_total):
    df = df_total.copy()
    df = df[df["AGLOMERADO"].isin(AGLOMERADOS_COD)]
    df["PERIODO"] = (df["ANO4"].astype(int).astype(str) + "-T" + df["TRIMESTRE"].astype(int).astype(str))
    df["ESTADO"] = pd.to_numeric(df["ESTADO"], errors="coerce")
    df = df.dropna(subset=["ESTADO"])

    resultado = []
    for periodo in df["PERIODO"].unique():
        for aglom in AGLOMERADOS_COD:
            df_temp = df[(df["PERIODO"] == periodo) & (df["AGLOMERADO"] == aglom)]

            if len(df_temp) == 0:
                continue

            poblacion_total = len(df_temp)
            ocupados = len(df_temp[df_temp["ESTADO"] == 1])
            desocupados = len(df_temp[df_temp["ESTADO"] == 2])
            inactivos = len(df_temp[df_temp["ESTADO"] == 3])

            pea = ocupados + desocupados

            tasa_desocupacion = (desocupados / pea * 100) if pea > 0 else 0
            tasa_actividad = ((pea / poblacion_total * 100) if poblacion_total > 0 else 0)
            tasa_empleo = ((ocupados / poblacion_total * 100) if poblacion_total > 0 else 0)

            resultado.append(
                {
                    "PERIODO": periodo,
                    "AGLOMERADO": NOMBRES_AGLOMERADOS[aglom],
                    "Tasa_Actividad": tasa_actividad,
                    "Tasa_Empleo": tasa_empleo,
                    "Tasa_Desocupacion": tasa_desocupacion,
                    "Ocupados": ocupados,
                    "Desocupados": desocupados,
                    "Inactivos": inactivos,
                    "PEA": pea,
                    "Poblacion_Total": poblacion_total,
                }
            )

    return pd.DataFrame(resultado)


def mostrar_tabla_tasas(df_total):
    df_tasas = calcular_tasas(df_total)
    if len(df_tasas) == 0:
        print("No hay datos disponibles para calcular tasas")
        return
    print("-" * 80)
    print(" TASAS LABORALES: CABA Y CONURBANO BONAERENSE (2016-2026)")
    print("-" * 80)
    print(df_tasas.to_string(index=False))
    print("-" * 80)


def grafico_tasa_generico(df_total, columna, titulo):
    df_tasas = calcular_tasas(df_total)
    if len(df_tasas) == 0:
        print("No hay datos disponibles")
        return
    pivot = df_tasas.pivot(
        index="PERIODO", columns="AGLOMERADO", values=columna
    )
    pivot.plot(kind="bar", figsize=(12, 6))
    plt.title(f"{titulo} (%)", fontsize=14, weight="bold")
    plt.xlabel("Período")
    plt.ylabel("Porcentaje (%)")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# 4. EVOLUCIÓN DE INGRESOS REALES

def mostrar_tabla_ingresos(df_total):
    df = df_total.copy()
    df = df[df["AGLOMERADO"].isin(AGLOMERADOS_COD)]
    df["P47T_real"] = pd.to_numeric(df["P47T_real"], errors="coerce")
    df = df[df["P47T_real"] > 0].dropna(subset=["P47T_real"])

    if len(df) == 0:
        print("No hay datos de ingresos disponibles")
        return

    df["PERIODO"] = (df["ANO4"].astype(int).astype(str) + "-T" + df["TRIMESTRE"].astype(int).astype(str))
    df["AGLOMERADO_NOMBRE"] = df["AGLOMERADO"].map(NOMBRES_AGLOMERADOS)

    df_grouped = (df.groupby(["PERIODO", "AGLOMERADO_NOMBRE"])["P47T_real"].agg([("Media", "mean"), ("Mediana", "median")]).reset_index())

    print("-" * 80)
    print("Evolución de ingresos reales (CABA vs CONURBANO)")
    print("-" * 80)
    print(df_grouped.to_string(index=False))
    print("-" * 80)


def grafico_ingreso_generico(df_total, metrica, titulo):
    df = df_total.copy()
    df = df[df["AGLOMERADO"].isin(AGLOMERADOS_COD)]
    df["P47T_real"] = pd.to_numeric(df["P47T_real"], errors="coerce")
    df = df[df["P47T_real"] > 0].dropna(subset=["P47T_real"])

    if len(df) == 0:
        print("No hay datos de ingresos disponibles")
        return

    df["PERIODO"] = (df["ANO4"].astype(int).astype(str) + "-T" + df["TRIMESTRE"].astype(int).astype(str))
    df["AGLOMERADO_NOMBRE"] = df["AGLOMERADO"].map(NOMBRES_AGLOMERADOS)

    if metrica == "mean":
        df_grouped = (df.groupby(["PERIODO", "AGLOMERADO_NOMBRE"])["P47T_real"].mean().reset_index())
    else:
        df_grouped = (df.groupby(["PERIODO", "AGLOMERADO_NOMBRE"])["P47T_real"].median().reset_index())

    df_grouped.columns = ["PERIODO", "AGLOMERADO_NOMBRE", "Valor"]
    pivot = df_grouped.pivot(index="PERIODO", columns="AGLOMERADO_NOMBRE", values="Valor")

    pivot.plot(kind="bar", figsize=(12, 6))
    plt.title(f"{titulo} (Pesos Reales)", fontsize=14, weight="bold")
    plt.xlabel("Período")
    plt.ylabel("Ingreso Real ($)")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 5. ANÁLISIS UNIVARIADO Y MEDIDAS CENTRALES

def analizar_univariado(df_total, variable):
    tasa = {
        "Ocupados": 1,
        "Desocupados": 2,
        "Inactivo": 3,
        "Menor de 10 años": 4,
    }

    df = df_total.copy()
    df["ESTADO"] = pd.to_numeric(df["ESTADO"], errors="coerce")
    df["AGLOMERADO"] = pd.to_numeric(df["AGLOMERADO"], errors="coerce")
    df = df.dropna(subset=["ESTADO", "AGLOMERADO"])

    df_filtrado = df[df["AGLOMERADO"].isin(AGLOMERADOS_COD)]
    df_limpio = df_filtrado[df_filtrado["ESTADO"] == tasa[variable]]

    if len(df_limpio) == 0:
        print(f"\n[!] No hay datos de '{variable}' para mostrar.")
        return

    df_limpio["PERIODO"] = (df_limpio["ANO4"].astype(int).astype(str) + "-T" + df_limpio["TRIMESTRE"].astype(int).astype(str))
    df_grouped = (df_limpio.groupby(["PERIODO", "AGLOMERADO"]).size().reset_index(name="TOTAL"))

    pivot = df_grouped.pivot(index="PERIODO", columns="AGLOMERADO", values="TOTAL").fillna(0)
    pivot = pivot.rename(columns=NOMBRES_AGLOMERADOS)

    pivot.plot(kind="bar", figsize=(12, 6))
    plt.title(f"Total de {variable} — CABA vs Conurbano", fontsize=12, weight="bold")
    plt.xlabel("Período")
    plt.ylabel("Cantidad de Registros Muestrales")
    plt.xticks(rotation=45)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


def estadisticas_resumen(df_total):
    print("-" * 50)
    print("Medidas de tendencia central y posición (INGRESOS)")
    print("-" * 50)

    df = df_total.copy()
    if "P47T_real" not in df.columns:
        print("ERROR: Ejecute primero el ajuste por inflación")
        return

    df["P47T_real"] = pd.to_numeric(df["P47T_real"], errors="coerce")
    df = df[df["P47T_real"] > 0].dropna(subset=["P47T_real"])

    if len(df) == 0:
        print(" No hay datos válidos de ingresos")
        return

    media = df["P47T_real"].mean()
    mediana = df["P47T_real"].median()
    p25 = df["P47T_real"].quantile(0.25)
    p75 = df["P47T_real"].quantile(0.75)

    print(f"Media:              ${media:,.2f}")
    print(f"Mediana:            ${mediana:,.2f}")
    print(f"Percentil 25 (P25): ${p25:,.2f}")
    print(f"Percentil 75 (P75): ${p75:,.2f}")
    print("-" * 50)


# 6. ANÁLISIS MULTIVARIADO

def analizar_multivariado(df_total, variable):
    mapa_estado = {
        "Ocupados": 1,
        "Desocupados": 2,
        "Inactivo": 3,
        "Menor de 10 años": 4,
    }
    mapa_sexo = {1: "Masculino", 2: "Femenino"}

    def clasificar_educacion(x):
        try:
            x = int(x)
        except Exception:
            return "NS/NR"
        if x in [1, 2, 3, 4, 5, 6]:
            return "Básico"
        if x in [7, 8, 9]:
            return "Superior"
        return "NS/NR"

    df = df_total.copy()
    for col in ["AGLOMERADO", "ANO4", "TRIMESTRE"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["AGLOMERADO", "ANO4", "TRIMESTRE"])
    df = df[df["AGLOMERADO"].isin(AGLOMERADOS_COD)]
    df["PERIODO"] = (df["ANO4"].astype(int).astype(str) + "-T" + df["TRIMESTRE"].astype(int).astype(str))

    if variable in mapa_estado:
        df["ESTADO"] = pd.to_numeric(df["ESTADO"], errors="coerce")
        df["CH04"] = pd.to_numeric(df["CH04"], errors="coerce")
        df = df.dropna(subset=["ESTADO", "CH04"])

        df = df[df["ESTADO"] == mapa_estado[variable]]
        df["SEXO"] = df["CH04"].map(mapa_sexo)

        if len(df) == 0:
            print(f"Sin datos suficientes para {variable}")
            return

        df_grouped = (df.groupby(["PERIODO", "AGLOMERADO", "SEXO"]).size().reset_index(name="TOTAL"))
        df_grouped["AGLOMERADO_TXT"] = df_grouped["AGLOMERADO"].map(NOMBRES_AGLOMERADOS)
        df_grouped["CATEGORIA"] = (df_grouped["AGLOMERADO_TXT"] + "-" + df_grouped["SEXO"])

        pivot = df_grouped.pivot(index="PERIODO", columns="CATEGORIA", values="TOTAL").fillna(0)
        pivot.plot(kind="bar", figsize=(12, 6))
        plt.title(f"{variable} — Distribución por Sexo y Aglomerado")
        plt.xticks(rotation=45)
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.show()
        return

    if variable.lower() == "educacion":
        df["NIVEL_SIMPLE"] = df["NIVEL_ED"].apply(clasificar_educacion)
        df_grouped = (df.groupby(["PERIODO", "AGLOMERADO", "NIVEL_SIMPLE"]).size().reset_index(name="TOTAL"))
        df_grouped["AGLOMERADO_TXT"] = df_grouped["AGLOMERADO"].map(NOMBRES_AGLOMERADOS)
        df_grouped["CATEGORIA"] = (df_grouped["AGLOMERADO_TXT"] + " - " + df_grouped["NIVEL_SIMPLE"])

        pivot = df_grouped.pivot(index="PERIODO", columns="CATEGORIA", values="TOTAL").fillna(0)
        pivot.plot(kind="bar", figsize=(12, 6))
        plt.title("Nivel Educativo — Comparación CABA vs Conurbano")
        plt.xticks(rotation=45)
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.show()


# 7. REGRESIÓN E IMPUTACIÓN DE INGRESOS

def modelacion_regresion(df_total):
    df = df_total.copy()

    for c in ["P47T", "CH06", "NIVEL_ED", "CH04", "PP04B_COD", "PP04D_COD"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df_con_ingreso = df.dropna(subset=["P47T", "CH06", "NIVEL_ED", "CH04"])
    df_sin_ingreso = df[
        df["P47T"].isna()
        & df["CH06"].notna()
        & df["NIVEL_ED"].notna()
        & df["CH04"].notna()
    ]

    if len(df_con_ingreso) == 0:
        print("Datos insuficientes para el entrenamiento")
        return

    X = df_con_ingreso[["CH06", "NIVEL_ED", "CH04", "PP04B_COD", "PP04D_COD"]].fillna(-1)
    y = df_con_ingreso["P47T"]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    modelo = LinearRegression()
    modelo.fit(x_train, y_train)

    y_pred = modelo.predict(x_test)

    print("-" * 60)
    print("Modelo de regresión para imputación de ingresos (AMBA)")
    print("-" * 60)
    print(f"Error MSE: {mean_squared_error(y_test, y_pred):,.2f}")
    print(f"R²: {r2_score(y_test, y_pred):.4f}")

    coef_df = pd.DataFrame(
        {
            "Variable": [
                "Edad (CH06)",
                "Nivel Educativo (NIVEL_ED)",
                "Sexo (CH04)",
                "Categoría Laboral (PP04B_COD)",
                "Rama de Actividad (PP04D_COD)",
            ],
            "Coeficiente": modelo.coef_,
        }
    )

    print("Coeficientes e influencia de las variables:")
    print("-" * 60)
    for v, c in zip(coef_df["Variable"], coef_df["Coeficiente"]):
        print(f"{v:<35} | {c:>18,.2f}")
    print("-" * 60)

    if len(df_sin_ingreso) > 0:
        X_imputar = df_sin_ingreso[["CH06", "NIVEL_ED", "CH04", "PP04B_COD", "PP04D_COD"]].fillna(-1)
        ingresos_imputados = modelo.predict(X_imputar)

        print(
            f"Registros imputados:{len(ingresos_imputados)} valores faltantes")
        print(f"Promedio imputado:${ingresos_imputados.mean():,.2f}")
        print(f"Rango:${ingresos_imputados.min():,.2f} - ${ingresos_imputados.max():,.2f}")
    print("-" * 60 )


# 8. MAPA GEOREFERENCIADO

def mapa_aglomerados():
    path_mapa = "Datos/aglomerados_eph.json"
    
    if not os.path.exists(path_mapa):
        print(f"No se encuentra el archivo de mapa en '{path_mapa}'")
        print("Asegurate de colocar 'aglomerados_eph.json' dentro de la carpeta 'Datos'")
        return

    try:
        mapa = gpd.read_file(path_mapa)
    except Exception as e:
        print("Error al cargar el archivo GeoJSON:", e)
        return

    # Buscar la columna que contiene los nombres de aglomerados
    col_nombre = None
    for col in mapa.columns:
        if col.lower() in ["aglomerado", "eph_nombre", "nombre", "nom_aglo"]:
            col_nombre = col
            break

    if col_nombre is None:
        col_nombre = mapa.columns[0]

    # Filtrar CABA / Conurbano
    patron = "Gran Buenos Aires|CABA|Ciudad Autonoma|Capital Federal|Buenos Aires"
    mapa_filtrado = mapa[mapa[col_nombre].astype(str).str.contains(patron, case=False, na=False)].copy()

    if len(mapa_filtrado) == 0:
        mapa_filtrado = mapa

    if mapa_filtrado.crs is None:
        mapa_filtrado = mapa_filtrado.set_crs("EPSG:4326")

    plt.close('all')
    
    # Crear figura con 2 subplots lado a lado
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: CABA (Rojo)
    sub_caba = mapa_filtrado[mapa_filtrado[col_nombre].astype(str).str.contains("CABA|Ciudad Autonoma|Capital", case=False, na=False)]
    if not sub_caba.empty:
        sub_caba.plot(ax=ax1, color="#FF5733", edgecolor="black", linewidth=1.2, alpha=0.8)
    ax1.set_title("Ciudad Autónoma de Buenos Aires (CABA)", fontsize=11, weight="bold")
    ax1.set_xlabel("Longitud")
    ax1.set_ylabel("Latitud")
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Subplot 2: Conurbano Bonaerense (Azul)
    sub_conurbano = mapa_filtrado[mapa_filtrado[col_nombre].astype(str).str.contains("Gran Buenos Aires|Conurbano", case=False, na=False)]
    if not sub_conurbano.empty:
        sub_conurbano.plot(ax=ax2, color="#3388FF", edgecolor="black", linewidth=1.2, alpha=0.8)
    ax2.set_title("Conurbano Bonaerense", fontsize=11, weight="bold")
    ax2.set_xlabel("Longitud")
    ax2.set_ylabel("Latitud")
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle("Aglomerados EPH: Comparativa Territorial AMBA", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.show(block=True)


# 9. MENÚ PRINCIPAL INTERACTIVO CON MATCH/CASE


def menu(df_total):
    menu_principal = """
---------------------------------------------------------------------
     SISTEMA DE ANÁLISIS DE DATOS EPH — CABA Y CONURBANO (2020-2026)     
---------------------------------------------------------------------
1. Cargar / Recargar archivos de datos
2. Tasas laborales
3. Evolución de ingresos reales
4. Análisis univariado por condición de actividad
5. Medidas de tendencia central y posición
6. Análisis multivariado (Sexo, Educación y Aglomerado)
7. Modelo de regresión e imputación de ingresos
8. Mapa georreferenciado
9. Salir
----------------------------------------------------------------------
"""

    menu_tasas = """
----------------------------------------------------------------------
TASAS LABORALES

A. Mostrar tabla completa de tasas
B. Gráfico de Tasa de Actividad
C. Gráfico de Tasa de Empleo
D. Gráfico de Tasa de Desocupación
E. Volver al menú principal
----------------------------------------------------------------------
"""

    menu_ingresos = """
----------------------------------------------------------------------
EVOLUCIÓN DE INGRESOS REALES

A. Mostrar tabla de ingresos reales (Media y Mediana)
B. Gráfico de Ingreso Promedio Real
C. Gráfico de Ingreso Mediano Real
D. Volver al menú principal
----------------------------------------------------------------------
"""

    menu_univariado = """
----------------------------------------------------------------------
ANÁLISIS UNIVARIADO

1. Ocupados
2. Desocupados
3. Inactivos
4. Menores de 10 años
5. Volver al menú principal
----------------------------------------------------------------------
"""

    menu_multivariado = """
----------------------------------------------------------------------
ANÁLISIS MULTIVARIADO

1. Ocupados por Sexo y Aglomerado
2. Desocupados por Sexo y Aglomerado
3. Inactivos por Sexo y Aglomerado
4. Menores de 10 años por Sexo y Aglomerado
5. Nivel Educativo por Aglomerado
6. Volver al menú principal
----------------------------------------------------------------------
"""

    while True:
        opcion = input(menu_principal).strip()

        match opcion:
            case "1":
                print("Cargando datos de 2020 a 2026...")
                df_total = cargar_datos()
                df_total = ajustar_por_inflacion(df_total)
                print(
                    f"Carga completa. Registros cargados: {len(df_total):,}"
                )

            case "2":
                while True:
                    sub_opcion = input(menu_tasas).strip().upper()
                    match sub_opcion:
                        case "A":
                            mostrar_tabla_tasas(df_total)
                        case "B":
                            grafico_tasa_generico(
                                df_total, "Tasa_Actividad", "Tasa de Actividad"
                            )
                        case "C":
                            grafico_tasa_generico(
                                df_total, "Tasa_Empleo", "Tasa de Empleo"
                            )
                        case "D":
                            grafico_tasa_generico(
                                df_total,
                                "Tasa_Desocupacion",
                                "Tasa de Desocupación",
                            )
                        case "E":
                            print("Volver al menú principal")
                            break
                        case _:
                            print("Opción incorrecta")

            case "3":
                while True:
                    sub_opcion = input(menu_ingresos).strip().upper()
                    match sub_opcion:
                        case "A":
                            mostrar_tabla_ingresos(df_total)
                        case "B":
                            grafico_ingreso_generico(
                                df_total, "mean", "Ingreso Promedio Real"
                            )
                        case "C":
                            grafico_ingreso_generico(
                                df_total, "median", "Ingreso Mediano Real"
                            )
                        case "D":
                            print("Volver al menú principal...")
                            break
                        case _:
                            print("Opción incorrecta")

            case "4":
                while True:
                    sub_opcion = input(menu_univariado).strip()
                    match sub_opcion:
                        case "1":
                            analizar_univariado(df_total, "Ocupados")
                        case "2":
                            analizar_univariado(df_total, "Desocupados")
                        case "3":
                            analizar_univariado(df_total, "Inactivo")
                        case "4":
                            analizar_univariado(df_total, "Menor de 10 años")
                        case "5":
                            print("Volver al menú principal")
                            break
                        case _:
                            print("Opción incorrecta")

            case "5":
                estadisticas_resumen(df_total)

            case "6":
                while True:
                    sub_opcion = input(menu_multivariado).strip()
                    match sub_opcion:
                        case "1":
                            analizar_multivariado(df_total, "Ocupados")
                        case "2":
                            analizar_multivariado(df_total, "Desocupados")
                        case "3":
                            analizar_multivariado(df_total, "Inactivo")
                        case "4":
                            analizar_multivariado(df_total, "Menor de 10 años")
                        case "5":
                            analizar_multivariado(df_total, "Educacion")
                        case "6":
                            print("Volver al menú principal")
                            break
                        case _:
                            print("Opción incorrecta")

            case "7":
                modelacion_regresion(df_total)

            case "8":
                mapa_aglomerados()

            case "9":
                print("Programa finalizado")
                break

            case _:
                print("Opción no válida. Ingrese un número del 1 al 9")

    return df_total


if __name__ == "__main__":
    generar_ipc_trimestral()
    df = pd.DataFrame()
    df = menu(df)