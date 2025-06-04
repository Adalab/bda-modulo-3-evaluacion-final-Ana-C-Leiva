#%%
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer

from scipy.stats import kstest
from scipy.stats import levene
from scipy.stats import mannwhitneyu
import scipy.stats as st

import seaborn as sns
import matplotlib.pyplot as plt

def conf():
    pd.set_option('display.max_columns', None) 


def read(root):
    df = pd.read_csv(root, sep=',', delimiter=None, header='infer', names=None, index_col=None, dtype=None)
    return df

def samples(df, number=5):
    return df.head(number), df.tail(number), df.sample(number)

def shape(df):
    print('Shape')
    print(df.shape)
    print(f"The number of rows is {df.shape[0]}, and the number of columns is {df.shape[1]}")
    print('---------------------')
    print('The columns are:')
    print(df.columns)

def left_merge(df_left, df_right, left_on, right_on):
    merge_left = df_left.merge( right=df_right, how='left',
                        left_on=left_on, right_on=right_on)
    return merge_left

def unique(df):
    coltypes = []
    coltypes = df.dtypes.tolist()
    
    for i in range(len(coltypes)):
        coltypes[i] = str(coltypes[i])
    
    unique_types = list(dict.fromkeys(coltypes))
    print(unique_types)
    return unique_types

def all_columns_types(df):
    print('These are the columns and the types')
    print(df.dtypes)
    return 

def description_by_type(df, types_list):
    for type in types_list:
        print('Description of variables of type', type)
        print(df.describe(include=[type]).T)
        print('-----------------------------------------')

def standard_name(df, rep1=".", rep2="_"):
    new_columns = {col: col.lower().replace(rep1, rep2) for col in df.columns}
    df.rename(columns = new_columns, inplace = True)
    print(df.columns)
    return

def lower_case(df, columns):
    for col in columns:
        df[col] = df[col].str.lower()
    return df[columns]

def cell_replacement(df, columns, rep1, rep2):
    for col in columns:
        df[col] = df[col].str.replace(rep1,rep2)
    return df[columns].head()

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

def col_numbers_null(df):
    number_col = df.select_dtypes(include=['float64']).columns
    number_col_with_nulls = number_col.intersection(nulls_list['var'])
    return number_col_with_nulls

def plot_numbers(df, columns):
    for col in list(columns):
        plt.figure()
        plt.hist(df[col].dropna(), bins=30, color='plum', edgecolor='black')
        plt.title(f'Histograma of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()

def iter_impute(df, columns):
    imputer_iter = IterativeImputer(max_iter = 10, random_state = 42)

    for col in columns:
        df[f'iter_{col}'] = imputer_iter.fit_transform(df[[col]])

def knn_impute(df, columns):
    imputer_knn = KNNImputer(n_neighbors=5)

    for col in columns:
        df[f'knn_{col}'] = imputer_knn.fit_transform(df[[col]])

def box_hist_cont(df, columns):
    for col in columns:
        fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 5))

        # boxplot with Seaborn using 'sns.boxplot()'
        sns.boxplot(y = col, 
                    data = df, 
                    width = 0.25, 
                    color = 'teal', 
                    ax = axes[0])

        # adding title using '.set_title()
        axes[0].set_title(f"Boxplot using Seaborn of {col} column")

        sns.histplot(x = col, 
                    data = df, 
                    color = "slateblue", 
                    kde = True, 
                    bins = 20 ,)

        # using 'plt.xlabel()' change the name of x-axes
        plt.xlabel(col)

        # using 'plt.ylabel()' change the name of y-axes
        plt.ylabel("Count")


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
                    color = 'teal',
                    ax = axes[0])

            # añadimos un título a esta primera gráfica usando el método '.set_title()
            axes[0].set_title(f"Boxplot using Seaborn of {col} column")
        
            # Gráfico de barras para la frecuencia relativa
            sns.barplot(x=col, 
                        y='Relative Frequency (%)', 
                        data=frequency_table,
                        palette = 'rocket',
                        hue = col,
                        ax=axes[1])
            axes[1].set_title(f'Relative Frequency for {col}')
            axes[1].set_xlabel(col)
            axes[1].set_ylabel('Relative Frequency (%)')
            axes[1].tick_params(axis='x', rotation=45)

            plt.tight_layout();

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

def duplicated_rows(df):
    duplicated_rows = df[df.duplicated()]
    number_rows = len(duplicated_rows)
    print('Duplicated rows:')
    print(number_rows)
    
    df_no_duplicates = df.drop_duplicates()
    return df_no_duplicates 

def fase_1():

    conf()

    df_loyalty = read('customer_loyalty_history.csv')
    df_activity = read('customer_flight_activity.csv')

    shape(df_loyalty)
    shape(df_activity)

    samples(df_loyalty)
    samples(df_activity)

    all_columns_types(df_loyalty)
    all_columns_types(df_activity)

    df_total =left_merge(df_activity, df_loyalty, 'Loyalty Number', 'Loyalty Number')
    shape(df_total) # same numbers of rows as df_left - ok, columns = c1 +c2 - 1 - ok
    samples(df_total)

    all_columns_types(df_total)
    standard_name(df_total, " ", "_")

    unique_types = unique(df_total)
    description_by_type(df_total,unique_types)

    # Name standarization
    standard_name(df_total, " ", "_")

    # Type standarization
    columns=['province','education','loyalty_card','marital_status','gender']
    for col in columns:
        print(df_total[col].unique())

    columns = ['education','marital_status','gender']
    lower_case(df_total, columns)

    columns=['education']
    cell_replacement(df_total,columns," ","_") # I decided not to change this aspect for the province

    columns = ['province','education','loyalty_card','marital_status','gender', 'province']
    for col in columns:
        print(df_total[col].unique()) # control of the changes

    # Treatment of nulls
    nulls_list = nulls(df_total,1,1) # among the relevant columns only 'salary' has null observations, with 25% high share
    nulls_list
    df_total['salary'].dtypes
    df = df_total.drop('cancellation_year', axis=1)
    df = df_total.drop('cancellation_month', axis=1)

    # Second tranformation of data - after checking it had negative values
        # salary0: all negative values become 0
        # salary1: all negative values become positive

    df_total['salary0'] = df_total['salary']
    df_total['salary0'] = df_total['salary0'].apply(lambda x: 0 if x < 0 else x)
    df_total['salary1'] = df_total['salary']
    df_total['salary1'] = df_total['salary1'].apply(lambda x: -x if x < 0 else x)

    columns = ['salary', 'salary0', 'salary1']
    iter_impute(df_total,columns)
    #knn_impute(df_total,columns) # with n_neighbors=2 not so different the two imputations - this one too time consuming

    plot_numbers(df_total, ['salary', 'salary0', 'salary1', 'iter_salary', 'iter_salary0', 'iter_salary1']) 
    # The share of observaction around the central values increases
        # the increase is different depending on the change I did on the negative salaries

    # Identify structure of data and look for outliers 

    # numeric contionuos variables
    columns =['distance', 'points_accumulated', 'iter_salary', 'salary', 'iter_salary0', 'salary0', 'iter_salary1', 'salary1']
    box_hist_cont(df_total, columns)

    # numeric discrete variables
    columns = ['flights_booked', 'month']
    count_discrete(df_total, columns)

    #descriptive statistics for numeric variables
    columns =['flights_booked', 'distance', 'points_accumulated', 'iter_salary', 'salary', 'iter_salary0', 'salary0', 'iter_salary1', 'salary1', 'year', 'month']
    describe_numeric(df_total, columns)

    # categorical variables and numeric with short spann
    columns = ['year','province','gender','education', 'loyalty_card']
    cat_plot(df_total,columns)

    columns = ['province','gender','education','loyalty_card']
    description_by_type(df_total[columns],['object'])

    columns = ['year','province','gender','education', 'loyalty_card']
    cat_plot(df_total,columns)

    columns = ['province','gender','education','loyalty_card']
    description_by_type(df_total[columns],['object'])

    # Duplicates
    df_final = duplicated_rows(df_total)
    duplicated_rows(df_final)

    return df_final


#FASE 2 - QUESTIONS
# Question 1 - ¿Cómo se distribuye la cantidad de vuelos reservados por mes durante el año?
# Grouping by year and month, I get the sum of all the flights_boked
def question1():
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
                palette='rocket',
                hue = 'month')
    axes[0].set_title('Share of Flights Booked per Month for 2017')
    axes[0].set_xlabel('Month')
    axes[0].set_ylabel('Share of Flights Booked')

    # Prepare and plot for 2018
    data_2018 = df_sorted[df_sorted['year'] == 2018].sort_values(by='share', ascending=False)
    sns.barplot(x='month', y='share',
                data=data_2018,
                order=data_2018['month'],
                ax=axes[1],
                palette='mako',
                hue = 'month')
    axes[1].set_title('Share of Flights Booked per Month for 2018')
    axes[1].set_xlabel('Month')
    axes[1].set_ylabel('Share of Flights Booked')

    # Final layout
    plt.tight_layout()
    plt.show()

# Question 2 - ¿Existe una relación entre la distancia 
# de los vuelos y los puntos acumulados por los cliente?

def question2():
    # Scatterplot
    sns.scatterplot(x="distance",
                    y="points_accumulated", 
                    data=df_final,
                    hue="loyalty_card",
                    palette = 'rocket',
                    marker="d", 
                    color="teal", 
                    s=1)

    # Labels and title
    plt.xlabel("Flight Distance")
    plt.ylabel("Points Accumulated")
    plt.title("Relationship between flight distance and accumulated points")

    # Remove top and right spines
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines["top"].set_visible(False)

    #Legend    
    plt.legend(title='Loyalty Card Status',
               title_fontsize=13,
               fontsize=10,
               handlelength=2.5,
               handleheight=1.5,
               markerscale=5)


    # Final layout
    plt.tight_layout()
    plt.show()

# Question 3 - ¿Cuál es la distribución de los clientes por provincia o estado?
def question3():
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

# Question 4 - ¿Cómo se compara el salario promedio entre los diferentes 
# niveles educativos de los clientes?
def question4():
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
        ax.bar_label(container, fmt='%.0f', label_type='edge', padding=3)


    plt.suptitle("Comparación de salario por nivel educativo")

    # añadimos el 'plt.tigth_layout()' para que se ajusten los elementos de la gráfica
    plt.tight_layout()
    plt.show()


# Question 5 - ¿Cuál es la proporción de clientes con diferentes tipos de tarjetas de fidelidad?
def question5():
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

    plt.tight_layout()
    plt.show()


# Question 6 - ¿Cómo se distribuyen los clientes según su estado civil y género?
def question6():
    df_marital_gender = df_final.groupby(['marital_status', 'gender'])['loyalty_number'].nunique().reset_index()
    df_marital_gender['share'] = df_marital_gender['loyalty_number'] / df_marital_gender.groupby('gender')['loyalty_number'].transform('sum')*100

    df_marital_gender_f = df_marital_gender[df_marital_gender['gender']=='female'].sort_values(by='share', ascending=False)
    ordered_marital_status = df_marital_gender_f['marital_status'].tolist()

    plt.figure(figsize=(12, 8))

    sns.barplot(x='marital_status', 
                y='share', 
                hue='gender', 
                palette = 'viridis',
                data=df_marital_gender,
                order=ordered_marital_status
                )
   
    plt.xlabel('Marital Status')
    plt.ylabel('Share of clients')
    plt.title('Share of clients by marital status and gender')

    plt.tight_layout()
    plt.show()


## FASE 3 - BONUS
# se busca evaluar si existen diferencias significativas en el número de vuelos 
# reservados según el nivel educativo de los clientes.

def fase_3():
    # Filter data - keep: 'flights_booked' and 'education'

    df_bonus = df_final.groupby(['loyalty_number','education'])['flights_booked'].sum().reset_index()
    df_bonus['flights_booked'].describe()

    box_hist_cont(df_bonus,['flights_booked'])

    # Descriptive analysis
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))

    sns.boxplot(x='education', 
                y='flights_booked', 
                data=df_bonus, 
                palette='rocket', 
                hue='education',
                ax=axes[0])

    axes[0].set_xlabel('Education Level')
    axes[0].set_ylabel('Flights booked')
    axes[0].set_title('Box Plot of flights booked by education level')
    axes[0].tick_params(rotation=45) # Tilt x-axis labels for readability
    sns.despine(top=True, right=True)


    ax = sns.barplot(x='education', 
                y='flights_booked', 
                hue='education', 
                data=df_bonus, 
                palette='rocket')

    axes[1].set_xlabel('Education Level')
    axes[1].set_ylabel('Averga flights booked')
    axes[1].set_title('Average flights booked by education level')
    axes[1].tick_params(rotation=45)
    sns.despine(top=True, right=True)

    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f', label_type='edge', padding=3)


    plt.suptitle("Comparison of flights booked by education level")

    plt.tight_layout()
    plt.show()

    for type in df_bonus['education'].unique():
        print(describe_numeric(df_bonus[df_bonus['education']==type],['flights_booked']))
        print(box_hist_cont(df_bonus[df_bonus['education']==type],['flights_booked']))

    # Two groups
    high_education =df_bonus[df_bonus['education'].isin(['college', 'master','doctor'])]
    low_education =df_bonus[df_bonus['education'].isin(['bachelor', 'high_school_or_below'])]

    #No zeros
    high_education_no_zeros = high_education[high_education['flights_booked']>0]
    low_education_no_zeros = low_education[low_education['flights_booked']>0]

    print(len(high_education))
    print(len(low_education))
    print(len(high_education_no_zeros))
    print(len(low_education_no_zeros))
    # OBS: 
        # With zeros:
            # due to the size of the subsamples, we can assume that the Central
            # limit theorem applies - and assume normality, regardless of the underlying
            # distribution
        # Without zeros
            # The same applies

    box_hist_cont(high_education,['flights_booked'])
    box_hist_cont(low_education,['flights_booked'])
    box_hist_cont(high_education_no_zeros,['flights_booked'])
    box_hist_cont(low_education_no_zeros,['flights_booked'])

    describe_numeric(high_education,['flights_booked'])
    describe_numeric(low_education,['flights_booked'])

    describe_numeric(high_education_no_zeros,['flights_booked'])
    describe_numeric(low_education_no_zeros,['flights_booked'])

    # Perform the Kolmogorov-Smirnov test
    ks_statistic, p_value = kstest(high_education['flights_booked'], 'norm')
    # Output the results
    print('The KS test for high education')
    print(f"K-S Test Statistic: {ks_statistic}")
    print(f"p-value: {p_value}")

    # Perform the Kolmogorov-Smirnov test
    ks_statistic, p_value = kstest(low_education['flights_booked'], 'norm')
    # Output the results
    print('The KS test for low education')
    print(f"K-S Test Statistic: {ks_statistic}")
    print(f"p-value: {p_value}")

    # Perform the Kolmogorov-Smirnov test
    ks_statistic, p_value = kstest(high_education_no_zeros['flights_booked'], 'norm')
    # Output the results
    print('The KS test for high education no zeros')
    print(f"K-S Test Statistic: {ks_statistic}")
    print(f"p-value: {p_value}")

    # Perform the Kolmogorov-Smirnov test
    ks_statistic, p_value = kstest(low_education_no_zeros['flights_booked'], 'norm')
    # Output the results
    print('The KS test for low education no zeros')
    print(f"K-S Test Statistic: {ks_statistic}")
    print(f"p-value: {p_value}")

    # OBS:
        # Based on the KS-test and the the two specifications of samples 
        # I reject null hypothesis that the samples are Normal

    # Tests for variance homogeneity - with zeros
    levene_statistic, p_value = levene(high_education['flights_booked'], low_education['flights_booked'])

    print(f"Levene's Test Statistic: {levene_statistic}")
    print(f"p-value: {p_value}")

    # Tests for variance homogeneity - without zeros
    levene_statistic, p_value = levene(high_education_no_zeros['flights_booked'], low_education_no_zeros['flights_booked'])

    print(f"Levene's Test Statistic: {levene_statistic}")
    print(f"p-value: {p_value}")

    # OBS:
        # With and without zeros - I do not reject the null hipothesis at common levels of confidence
        # That means that I cannot reject the null hipothesis and that there is no statistical difference between the samples' variances

    # OBS:
        # Based on the Normality test, I should run a non parametric test of means,
        # But it is a promosing feature that the variances do not seem to be different in any of the sample specifications
        # Due to the CLT I will also conduct conduct parametric estimations (big sample)

    # Statistical tests
    #### Hypothesis
    # - H0: average flights booked (high_educated) = average flights_booked (lower_educated) 
    # - H1: average flights booked (high_educated) != average flights_booked (lower_educated) 

    result = mannwhitneyu(high_education['flights_booked'], low_education['flights_booked'], alternative='two-sided')
    print(result)

    result = mannwhitneyu(high_education_no_zeros['flights_booked'], low_education_no_zeros['flights_booked'], alternative='two-sided')
    print(result)

    # OBS:
        # For both cases of specification of the samples, I do not reject the null hypothesis (equall averages)
        # Conclussion: the means do not seam to differ in those two samples

    # Parametric
    result = st.ttest_ind(high_education['flights_booked'], low_education['flights_booked'], equal_var = True)
    print(result)

    result = st.ttest_ind(high_education_no_zeros['flights_booked'], low_education_no_zeros['flights_booked'], equal_var = True)
    print(result)

    # OBS:
        # Based on the parametric testing
            # Sample including zeros:
                # I only reject the null hypothesis at 10% of significance level
                # But at higher levels of confidence, such as 95% or more, it is not possible to reject the null hypothesis
            # Sample without zeros:
                # It is not possible to reject the null hipothesis to any common significance level
                # 
            # General conclussion: at 95% confidence level, it is not possible to reject that the means are the same for both samples  

#%%
df_final = fase_1()
#OBS:
# FLIGHTS BOOKED
    # MODE = 0 - Flights booked seem to have a mass concentration around 0, several without bookings
        # Could this be because there are months in which clients do not book flights? CHECK
    # The median is on the other hand in 1 while the median is several br flights bigger
    # The 50% central is between 0 and 8 flights
    # 50% of the sample is has 0 or 1 flight
    # There are not many outliers in this variable, an it seems not be me much forther than IQR*1.5

# DISTANCE
    # Has the same characteristic as the revious variable, with a concentration in 0 km
        # check the same as before
    # Presents more outliers but relatively close to the IQR*1.5

# POINTS ACCUMULATED
    # similar to DISTANCE
    # The central 50% is between 0 and 239 points, really close to 0, which is the mode
        # The zero value is going to have a lot of leverage in the estimations

# IMPUTED SALLARY
    # High concentration around the central values
        # Careful, this could affect the estimations and descriptive variables
    # MODE: 79.268, MEAN: 79.268, MEDIAN: 79268 --> THE SAME
    # The IMPUTED SALARY variable concentrates a big part of the estimation around the mode
        # CONTROL THE IMPUTED VALUES - are they all the same?

#SALARY
    # Negative income cannot be right! CHECK THIS
    # Two possibilities to treat the negative sallaries:
        # Impute them equall to 1
        # Assume that it was a typo, and tranform them into positives
        # Introduce both solutions and see the results
        # MODE: 101933, MEAN:  79268, MEDIAN: 73479 --> they differe

# IMPUTED 0 and 1
    # I will do the estimations with both variables and see the differences
    # There seems to be slight differences when it comes to estimation depending
        #  on the type of imputation I do before with the negative numbers
    # This columns are important to check after the robustness of the calculations

# OBS:
# YEAR
    # There are two years - exactly half and half

# PROVINCE
    # High concentration in the first three provinces - with two digits each
    # 11 unique values
# GENDER
    # Balanced among genders - almost 50-50
    # 2 unique values

# EDUCATION
    # High concentration among clients with bachelors degree more than 60%
    # The second category has a 1/4 of the clients
    # 5 unique values

# LOYALTY CARD
    # Almost half (45%) of the clients have the star loyalty card
    # 3 unique values

# FASE 2
# Question 1
question1()
# OBS:
    # The months with higher levels of flights booked are July, June and auguts for both years
    # The months with lowest demand are February, and January - again for both years

# Question 2
question2()
# OBS: 
    # There seems to be a clear linnear correlation between distance and accumulated points
    # This positive correlation increases the higher the loyalty level is, for the most
        # That is why there are three different lines with distinct collors
    # There is also a fourth line that is probably a combination of all the loyalty groups
        # This might be the result of some customers redeeming the points, cashing them, or loosing them because of tim
        # This assumption should be checked 

# Question 3
question3()
# OBS:
    # The three provinces with the highest share of customers are:
        # Ontario, British Columbia and Quebec
        # All with two digits
    # The resto of the provinces are more similar to each other when it comes to the share of clients

#Question 4
question4()
# OBS:
    # In line with the theory about human capital and wages:
        # The group with a higher mean salary is the one with the highest education level
        # The group with the lowest mean income is the group with lower education level
    # Dispersion:
        # The group with the highest dispersion of data is the one with doctorate
        # The group with lowest dispersion is the group with education level of 

# Question 5
question5()
# OBS:
    # The Fidelity group whith the highest share of clients is 'Star', around 45%
    # The group with less is 'Aurora'
        # This result together with the one from Q2 might suggest that this is the most exclusive category 

# Question 6
question6()
#OBS:
    # The share of marital status is pretty similar across gender groups.
    # Probably there is no statistically different difference among those groups share
        # But I have not checked this yet

## FASE 3 - BONUS
fase_3()
# OBS
    # First figure seems to show an accumulation of observations around clients with no flights booked
        # This pattern is repeated across all the groups
    # By inspecting the mean and the distribution of the variable
        # 'flights_booked' at first glance they seem highly similar
        # similar means and IQR, situated both at similar levels
        # The most differences are in the magnitude of the outliers 
        # and a bit less on the SD
    #  I created two groups - one including the zeros and other one without them
        # In order to check the stability of the estimations
        # With zeros:
            # due to the size of the subsamples, we can assume that the Central
            # limit theorem applies - and assume normality, regardless of the underlying
            # distribution
        # Without zeros
            # The same applies
    # Test for normality:
        # Based on the KS-test and the the two specifications of samples 
            # I reject the null hypothesis that the samples are Normally Distributed
    # Equality of variances:
        # With and without zeros - I do not reject the null hipothesis (at usual levels of confidence)
        # That means that I cannot reject the null hipothesis and that there is no statistical difference 
            # between the samples' variances

        # Based on the Normality test, I should run a non parametric test of means,
        # But it is a promosing feature that the variances do not seem to be different in any of the sample specifications
        # Due to the CLT I will also conduct conduct parametric estimations (due to the samples being big)

    # Statistical tests
    #### Hypothesis
    # - H0: average flights booked (high_educated) = average flights_booked (lower_educated) 
    # - H1: average flights booked (high_educated) != average flights_booked (lower_educated) 

    # Non-Parametric
        # For both cases of specification of the samples, I do not reject the null hypothesis (equall averages)
        # Conclussion: the means do not seam to differ in those two samples
    # Parametric
        # Sample including zeros:
            # I only reject the null hypothesis at 10% of significance level
            # But at higher levels of confidence, such as 95% or more, it is not possible to reject the null hypothesis
        # Sample without zeros:
            # It is not possible to reject the null hipothesis to any common significance level
        # General conclussion: at 95% confidence level, it is not possible to reject that the means are the same for both samples  
    # The parametric and non parametric estimations lead to the same conclussion - except for one of them that is not that relliable given the sample size, the assumptions tests, and the confidence level
    
    #FINAL CONCLUSSION: There is no evidence to rull out that the means are equall.

# %%
