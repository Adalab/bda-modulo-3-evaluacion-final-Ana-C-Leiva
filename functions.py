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

df_total['salary0'] = df_total['salary']
df_total['salary0'] = df_total['salary0'].apply(lambda x: 0 if x < 0 else x)
df_total['salary1'] = df_total['salary']
df_total['salary1'] = df_total['salary1'].apply(lambda x: -x if x < 0 else x)

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

#%% Analysis of categorical variables with short span
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
#%% Duplicates
def duplicated_rows(df):
    duplicated_rows = df[df.duplicated()]
    number_rows = len(duplicated_rows)
    print('Duplicated rows:')
    print(number_rows)
    
    df_no_duplicates = df.drop_duplicates()
    return df_no_duplicates 

df_final = duplicated_rows(df_total)

duplicated_rows(df_final)
#%% FASE 2 - Questions
# 1 Grouping by year and month, I get the sum of all the flights_boked
grouped_df = df_final.groupby(['year', 'month'])['flights_booked'].sum().reset_index()
grouped_df['share'] = grouped_df['flights_booked'] / grouped_df.groupby('year')['flights_booked'].transform('sum')*100

# barplot for both years
plt.figure(figsize=(12, 8))
sns.barplot(x='month', y='share', hue='year', data=grouped_df)
# Plot 1 - ordered by month
plt.xlabel('Month')
plt.ylabel('Share of Flights Booked')
plt.title('Share of Flights Booked per Month for 2017 and 2018')
plt.show()

# Plot 2 - ordered by share
# Sort the data by year and share
df_sorted = grouped_df.sort_values(by=['year', 'share'], ascending=[True, False])

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

# Prepare and plot for 2017
data_2017 = df_sorted[df_sorted['year'] == 2017].sort_values(by='share', ascending=False)
sns.barplot(x='month', y='share',
            data=data_2017,
            order=data_2017['month'],
            ax=axes[0],
            palette='rocket')
axes[0].set_title('Share of Flights Booked per Month for 2017')
axes[0].set_xlabel('Month')
axes[0].set_ylabel('Share of Flights Booked')

# Prepare and plot for 2018
data_2018 = df_sorted[df_sorted['year'] == 2018].sort_values(by='share', ascending=False)
sns.barplot(x='month', y='share',
            data=data_2018,
            order=data_2018['month'],
            ax=axes[1],
            palette='mako')
axes[1].set_title('Share of Flights Booked per Month for 2018')
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Share of Flights Booked')

# Final layout
plt.tight_layout()
plt.show()

# %%
# Question 2 - ¿Existe una relación entre la distancia 
# de los vuelos y los puntos acumulados por los cliente?
sns.regplot(x = "distance", 
            y = "points_accumulated", 
            data = df_final, 
            marker = "d", 
            line_kws = {"color": "black", "linewidth": 1}, # cambiamos el color y el grosor de la linea de tendencia
            scatter_kws = {"color": "teal", "s": 1} # cambiamos el color y el tamaño de los puntos del scaterplot
            )

# cambiamos los nombres de los ejes como hemos estado haciendo hasta ahora
plt.xlabel("Flight Distance")
plt.ylabel("Points Accumulated")

# ponemos título a la gráfica
plt.title("Relationship between flight distance and accumulated points")

# quitamos la linea de arriba y de la derecha
plt.gca().spines['right'].set_visible(False) # quitamos la línea de la derecha
plt.gca().spines["top"].set_visible(False) # quitamos la línea de arriba;

#%%
df_final.columns
# %%
# Question 3 - ¿Cuál es la distribución de los clientes por provincia o estado?
df_province = df_final.groupby('province')['loyalty_number'].nunique().reset_index()
df_province = df_province.rename(columns={'loyalty_number': 'clients_count'}) 
df_province['share'] = df_province['clients_count'] / df_province['clients_count'].sum()*100
df_province = df_province.sort_values(by='share', ascending=False)

# barplot for both years
sns.barplot(x='province', y='share', hue='province', data=df_province, palette='rocket')
# Plot 1 - ordered by month
plt.xlabel('province')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Share of Clients')
plt.title('Share of clients by Province')
plt.show()
# %%
# Question 4 - ¿Cómo se compara el salario promedio entre los diferentes 
# niveles educativos de los clientes?
df_salary_education = df_final.groupby(['education'])[['iter_salary', 'iter_salary0','iter_salary1']].mean().reset_index()
df_salary_education = df_salary_education.sort_values(by='iter_salary', ascending=False).round(0)
ordered_education_levels = df_salary_education['education'].tolist()

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))

sns.boxplot(x='education', 
            y='iter_salary1', 
            data=df_final, 
            palette='rocket', hue='education',
            order=ordered_education_levels,
            ax=axes[0])

axes[0].set_xlabel('Education Level')
axes[0].set_ylabel('Salary')
axes[0].set_title('Box Plot of Salary by Education Level')
axes[0].tick_params(rotation=45) # Tilt x-axis labels for readability
sns.despine(top=True, right=True)


ax = sns.barplot(x='education', 
            y='iter_salary1', 
            hue='education', 
            data=df_salary_education, 
            palette='rocket')

axes[1].set_xlabel('Education category')
axes[1].set_ylabel('Average salary')
axes[1].set_title('Average salary by education level')
axes[1].tick_params(rotation=45)
sns.despine(top=True, right=True)

for container in ax.containers:
    ax.bar_label(container, fmt='%.2f%%', label_type='edge', padding=3)


plt.suptitle("Comparación de salario por nivel educativo")

# añadimos el 'plt.tigth_layout()' para que se ajusten los elementos de la gráfica
plt.tight_layout()

# %%
# Question 5 - ¿Cuál es la proporción de clientes con diferentes tipos de tarjetas de fidelidad?
df_loyalty_level = df_final.groupby('loyalty_card')['loyalty_number'].nunique().reset_index()
df_loyalty_level = df_loyalty_level.rename(columns={'loyalty_number': 'clients_count'}) 
df_loyalty_level['share'] = df_loyalty_level['clients_count'] / df_loyalty_level['clients_count'].sum()*100
df_loyalty_level = df_loyalty_level.sort_values(by='share', ascending=False)
df_loyalty_level

ax = sns.barplot(x='loyalty_card', 
            y='share', 
            hue='loyalty_card', 
            data=df_loyalty_level, 
            palette='rocket')

plt.xlabel('Loyalty Category')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Share of Clients')
plt.title('Share of clients by Loyalty Category')
sns.despine(top=True, right=True)

for container in ax.containers:
    ax.bar_label(container, fmt='%.2f%%', label_type='edge', padding=3)


plt.show()
# %%
# Question 6 - ¿Cómo se distribuyen los clientes según su estado civil y género?
df_marital_gender = df_final.groupby(['marital_status', 'gender'])['loyalty_number'].nunique().reset_index()
df_marital_gender['share'] = df_marital_gender['loyalty_number'] / df_marital_gender.groupby('gender')['loyalty_number'].transform('sum')*100

df_marital_gender_f = df_marital_gender[df_marital_gender['gender']=='female'].sort_values(by='share', ascending=False)
ordered_marital_status = df_marital_gender_f['marital_status'].tolist()

#%%
# barplot for both years
plt.figure(figsize=(12, 8))
sns.barplot(x='marital_status', 
            y='share', 
            hue='gender', 
            palette = 'viridis',
            data=df_marital_gender,
            order=ordered_marital_status
            )
# Plot 1 - ordered by month
plt.xlabel('Marital Status')
plt.ylabel('Share of clients')
plt.title('Share of clients by marital status and gender')
plt.show()
# %%
## FASE 3 - BONUS
# se busca evaluar si existen diferencias significativas en el número de vuelos 
# reservados según el nivel educativo de los clientes.

# Filter data - keep: 'flights_booked' and 'education'
