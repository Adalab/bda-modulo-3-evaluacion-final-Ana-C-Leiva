
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

unique = unique(df_total)
description(df_total,unique)

columns=['province','education','loyalty_card','marital_status','gender']
for col in columns:
    print(df_total[col].unique())

columns = ['education','marital_status','gender']
lower_case(df_total, columns)
columns=['education']
cell_replacement(df_total,columns," ","_")

# Nulls
nulls(df_total,1,1) # among the relevant columns only 'salary' has null observations, with 25% high share

df = df_total.drop('cancellation_year', axis=1)
df = df_total.drop('cancellation_month', axis=1)

nulls_list = nulls(df_total,1,1)
nulls_list
df_total['salary'].dtypes

numbers_null = col_numbers_null(df_total)

plot_numbers(df_total,numbers_null)

# Second tranformation of data - after checking it had negative values
    # salary0: all negative values become 0
    # salary1: all negative values become positive

df_total['salary0'] = df_total['salary']
df_total['salary0'] = df_total['salary0'].apply(lambda x: 0 if x < 0 else x)
df_total['salary1'] = df_total['salary']
df_total['salary1'] = df_total['salary1'].apply(lambda x: -x if x < 0 else x)

columns = ['salary', 'salary0', 'salary1']
iter_impute(df_total,columns)
#knn_impute(df_total,columns) # with n_neighbors=2 not so different the two imputations

plot_numbers(df_total, ['iter_salary','salary'])
# The share of observaction around the central values increases

# Data distribution and outliers
columns =['distance', 'points_accumulated', 'iter_salary', 'salary']
box_hist_cont(df_total, columns)

columns = ['flights_booked', 'month']
count_discrete(df_total, columns)

columns =['flights_booked', 'distance', 'points_accumulated', 'iter_salary', 'salary', 'year']
describe_numeric(df_total, columns)

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

# MONTH
columns = ['year','province','gender','education', 'loyalty_card']
cat_plot(df_total,columns)

columns = ['province','gender','education','loyalty_card']
description_by_type(df_total[columns],['object'])
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

# FASE 2 - Questions

# ¿Cómo se distribuye la cantidad de vuelos reservados por mes durante el año?
grouped_df = df_final.groupby(['year', 'month'])['flights_booked'].sum().reset_index()