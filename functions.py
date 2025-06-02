#%% Import of libraries
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer

import seaborn as sns
import matplotlib.pyplot as plt

# Configuración
# -----------------------------------------------------------------------
pd.set_option('display.max_columns', None) # para poder visualizar todas las columnas de los DataFrames
#%% Lectura de datos
def read(root):
    df = pd.read_csv(root, sep=',', delimiter=None, header='infer', names=None, index_col=None, dtype=None)
    return df

df_loyalty = read('customer_loyalty_history.csv')
df_activity = read('customer_flight_activity.csv')

#%% Exploracion general del dataset
def samples(df, number=5):
    return df.head(number), df.tail(number), df.sample(number)

samples(df_loyalty)
samples(df_activity)

#%%
def shape(df):
    print('Shape')
    print(df.shape)
    print(f"The number of rows is {df.shape[0]}, and the number of columns is {df.shape[1]}")
    print('---------------------')
    print('The columns are:')
    print(df.columns)

shape(df_loyalty)
shape(df_activity)

#%% Concatenate and Merge Dataframes -- IMPO: define dframes as a list
def left_merge(df_left, df_right, left_on, right_on):
    merge_left = df_left.merge( right=df_right, how='left',
                        left_on=left_on, right_on=right_on)
    return merge_left

df_total =left_merge(df_activity, df_loyalty, 'Loyalty Number', 'Loyalty Number')
shape(df_total)
#%% 
def unique(df):
    coltypes = []
    coltypes = df.dtypes.tolist()
    
    for i in range(len(coltypes)):
        coltypes[i] = str(coltypes[i])
    
    unique_types = list(dict.fromkeys(coltypes))
    print(unique_types)
    return unique_types

unique = unique(df_total)

#%%
def all_columns_types(df):
    print('These are the columns and the types')
    print(df.dtypes)
    return 

all_columns_types(df_loyalty)
all_columns_types(df_activity)

#%% Estadísticas Básicas Variables Numéricas y Frecuencias Variables Categóricas
def description_by_type(df, types_list):
    for type in types_list:
        print('Description of variables of type', type)
        print(df.describe(include=[type]).T)
        print('-----------------------------------------')

description_by_type(df_total,unique)

#%% Transformación de tipo de datos y Estandarización de nombre de variables
def standard_name(df, rep1=".", rep2="_"):
    new_columns = {col: col.lower().replace(rep1, rep2) for col in df.columns}
    df.rename(columns = new_columns, inplace = True)
    print(df.columns)
    
    return


standard_name(df_total, " ", "_")

#%% Limpieza y Transformación Dependiendo de Resultado Anterior
def lower_case(df, columns):
    for col in columns:
        df[col] = df[col].str.lower()
    return df[columns]

columns = ['education','loyalty_card','marital_status','gender']
lower_case(df_total, columns)

#%%
def cell_replacement(df, columns, rep1, rep2):
    for col in columns:
        df[col] = df[col].str.replace(rep1,rep2)
    return df[columns].head()

columns=['education']
cell_replacement(df_total,columns," ","_")

#%% Identificación y Gestión de Nulos
def nulls(df,count=0,share=0):
    
    nulls = df.isnull().sum()
    nulls_share = df.isnull().sum()/df.shape[0]*100
    with_nulls_share = nulls_share[nulls_share > 0]
    
    if count == 1:
        print('Count of nulls')
        with_nulls = nulls[nulls > 0]
        print (with_nulls.sort_values(ascending=False))
        print('-----------------------------------')
    
    if share == 1:
        print('share of nulls')
        print (with_nulls_share.sort_values(ascending=False))
    
    nulls_list = with_nulls_share.to_frame(name='perc_nulos').reset_index().rename(columns={'index': 'var'})
    
    return nulls_list

nulls_list = nulls(df_total,1,1)
nulls_list
df_total['salary'].dtypes
#%%
def col_numbers_null(df):
    number_col = df.select_dtypes(include=['float64']).columns
    number_col_with_nulls = number_col.intersection(nulls_list['var'])
    return number_col_with_nulls

numbers_null = col_numbers_null(df_total)

#%%
def plot_numbers(df, columns):
    for col in list(columns):
        plt.figure()
        plt.hist(df[col].dropna(), bins=30, color='green', edgecolor='black')
        plt.title(f'Histograma of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()

plot_numbers(df_total,numbers_null)
#%%
def iter_impute(df, columns):
    imputer_iter = IterativeImputer(max_iter = 10, random_state = 42)

    for col in columns:
        df[f'iter_{col}'] = imputer_iter.fit_transform(df[[col]])

def knn_impute(df, columns):
    imputer_knn = KNNImputer(n_neighbors=5)

    for col in columns:
        df[f'knn_{col}'] = imputer_knn.fit_transform(df[[col]])

columns = ['salary', 'salary0', 'salary1']
iter_impute(df_total,columns)
#knn_impute(df_total,columns)

plot_numbers(df_total, ['salary', 'salary0', 'salary1', 'iter_salary', 'iter_salary0', 'iter_salary1']) 

# %% Identify structure of data and look for outliers
#numeric variables
def box_hist_cont(df, columns):
    for col in columns:
        fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 5))

        # añadimos un boxplot creado con Seaborn usando el método 'sns.boxplot()'
        sns.boxplot(y = col, 
                    data = df, 
                    width = 0.25, 
                    color = "turquoise", 
                    ax = axes[0])

        # añadimos un título a esta primera gráfica usando el método '.set_title()
        axes[0].set_title(f"Boxplot using Seaborn of {col} column")

        sns.histplot(x = col, 
                    data = df, 
                    color = "violet", 
                    kde = True, 
                    bins = 20 )

        # usando 'plt.xlabel()' cambiamos el nombre del eje x
        plt.xlabel(col)

        # usando el método 'plt.ylabel()' cambiamos el nombre del eje y
        plt.ylabel("Count")

columns =['flights_booked', 'distance', 'points_accumulated', 'iter_salary', 'salary', 'iter_salary0', 'salary0', 'iter_salary1', 'salary1']
box_hist_cont(df_total, columns)

#%%
def count_discrete(df,columns):
    for col in columns:
            absolute_frequency = df[col].value_counts()
            relative_frequency = df[col].value_counts(normalize=True) * 100

            # creamos un DataFrame para mostrar ambas tablas
            frequency_table = pd.DataFrame({col: absolute_frequency.index,
                                            'Absolute Frequency': absolute_frequency.values,
                                            'Relative Frequency (%)': relative_frequency.values
                                            })
            
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))

            # añadimos un boxplot creado con Seaborn usando el método 'sns.boxplot()'
            sns.boxplot(y = col, 
                    data = df, 
                    width = 0.25, 
                    color = "turquoise", 
                    ax = axes[0])

            # añadimos un título a esta primera gráfica usando el método '.set_title()
            axes[0].set_title(f"Boxplot using Seaborn of {col} column")
        
            # Gráfico de barras para la frecuencia relativa
            sns.barplot(x=col, y='Relative Frequency (%)', data=frequency_table, ax=axes[1])
            axes[1].set_title(f'Relative Frequency for {col}')
            axes[1].set_xlabel(col)
            axes[1].set_ylabel('Relative Frequency (%)')
            axes[1].tick_params(axis='x', rotation=45)

            plt.tight_layout();

columns = ['flights_booked', 'month']
count_discrete(df_total, columns)

#%%
def describe_numeric(df,columns):
    summary_df = pd.DataFrame()

    for col in columns:        
        mode = df[col].mode()[0]
        max = df[col].max() 
        min = df[col].min()
        iqr = np.percentile(df[col], 75) - np.percentile(df[col], 25)
        
        summary = df[col].describe()
        
        summary['mode'] = mode
        summary['max'] = max
        summary['min'] = min
        summary['iqr'] = iqr
    
        summary_df[col] = summary

    print(summary_df[columns])

columns =['flights_booked', 'distance', 'points_accumulated', 'iter_salary', 'salary', 'iter_salary0', 'salary0', 'iter_salary1', 'salary1', 'year']
describe_numeric(df_total, columns)

#%% Analisis of categorical variables with short span
# Crear un subplot con 2 filas y 1 columna
def cat_plot(df, columns):
    for col in columns:
        absolute_frequency = df[col].value_counts()
        relative_frequency = df[col].value_counts(normalize=True) * 100
        print(relative_frequency)  
        # creamos un DataFrame para mostrar ambas tablas
        frequency_table = pd.DataFrame({col: absolute_frequency.index,
                                        'Absolute Frequency': absolute_frequency.values,
                                        'Relative Frequency (%)': relative_frequency.values
                                        })
        
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))

        # Gráfico de barras para la frecuencia absoluta
        sns.barplot(x=col, 
                    y='Absolute Frequency', 
                    data=frequency_table, ax=axes[0])
        axes[0].set_title(f'Absolute Frequency for {col}')
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('Absolute Frequency')
        axes[0].tick_params(axis='x', rotation=45)

        # Gráfico de barras para la frecuencia relativa
        sns.barplot(x=col, y='Relative Frequency (%)', data=frequency_table, ax=axes[1])
        axes[1].set_title(f'Relative Frequency for {col}')
        axes[1].set_xlabel(col)
        axes[1].set_ylabel('Relative Frequency (%)')
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout();      

columns = ['year','province','gender','education', 'loyalty_card']
cat_plot(df_total,columns)

columns = ['province','gender','education','loyalty_card']
description_by_type(df_total[columns],['object'])
#%%