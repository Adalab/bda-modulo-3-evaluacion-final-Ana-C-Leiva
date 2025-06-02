
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

columns = ['salary']
iter_impute(df_total,columns)
#knn_impute(df_total,columns) # with n_neighbors=2 not so different the two imputations

plot_numbers(df_total, ['iter_salary','salary'])
# The share of observaction around the central values increases

# Data distribution and outliers
columns =['flights_booked', 'distance', 'points_accumulated', 'iter_salary', 'salary']
box_hist_cont(df_total, columns)

columns = ['flights_booked', 'month']
count_discrete(df_total, columns)

columns =['flights_booked', 'distance', 'points_accumulated', 'iter_salary', 'salary']
describe_numeric(df_total, columns)

columns = ['year','province','gender','education']
cat_plot(df_total,columns)

