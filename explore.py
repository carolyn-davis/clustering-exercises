#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 09:24:47 2021

@author: carolyndavis
"""
# =============================================================================
#                 CLUSTERING: EXPLORATION EXERCISES
# =============================================================================
# =============================================================================
# Exercises I - Required
# =============================================================================
# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Wrangling
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Statistical Tests
import scipy.stats as stats

# Visualizing
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from sklearn.model_selection import learning_curve

import wrangle as w

# =============================================================================
# 
# Our Telco scenario continues:
# 
# As a customer analyst for Telco, you want to know who has spent the most money with
#  the company over their lifetime. You have monthly charges and tenure, so you think you
#  will be able to use those two attributes as features to estimate total charges.
#  You need to do this within an average of $5.00 per customer.
# 
# In these exercises, you will run through the stages of exploration as you continue
#  to work toward the above goal.
# 
# Do your work in a notebook named explore.ipynb. In addition, you should create
# a file named explore.py that contains the following functions for exploring your
#  variables (features & target).
# =============================================================================

df = w.get_mall_customers('SELECT * FROM customers')
train, validate, test = w.train_validate_test_split(df)



df = train

df.head()
# =============================================================================
# 1.) Make sure to perform a train, validate, test split before and use only
#  your train dataset to explore the relationships between independent variables
#  with other independent variables or independent variables with your target
#  variable.
# =============================================================================
fig, axs = plt.subplots(1, 3, figsize=(15, 7))

for ax, col in zip(axs, df.select_dtypes('number')):
    df[col].plot.hist(ax=ax, title=col, ec='black')


df.gender.value_counts().plot.barh()


#Takeaways:
    #spending score is ~ normal
    #age + annual income have a long tail on the right -- i.e. they are right skewed
    #more female observations than male
    #Does spending score differ across gender?
    #Viz gender against spending score
    #Stats test to confirm
sns.violinplot(data=df, y='spending_score', x='gender')


sns.boxplot(data=df, y='spending_score', x='gender')
plt.title("Is there a difference in spending score for\nmale vs. female customers?")



sns.barplot(data=df, y='spending_score', x='gender')   

#Takeaways:
    # Seems like there's not much difference in spending score.
    
    
    # T-test:one-tailed or two-tailed? 2 tailed b/c we are looking for any difference in means
    #one-sample or two-sample? 2 sample b/c we're looking at the average spending score of 2 separate samples
    
    #Levene's Test: test for equal variance
    #H0: here is no difference in spending score variance between the two samples
    #Ha: there is a difference in spending score variance between the two samples

    


stats.levene(
    df.query('gender == "Male"').spending_score,
    df.query('gender == "Female"').spending_score,
)

#output: LeveneResult(statistic=0.10928566487557842, pvalue=0.7415451203905439)
#A high pvalue (.74) means we fail to reject the null hypothesis.

#
stats.ttest_ind(
    df.query('gender == "Male"').spending_score,
    df.query('gender == "Female"').spending_score,
    equal_var=True,
)
# Ttest_indResult(statistic=0.24250945188004078, pvalue=0.8088064406384925)
#We conclude there is no significant difference in spending score between Males and Females (p = .809).

# Conclusion:
# Is there a relationship between spending score and annual income?
#1.)Viz annual_income by spending_score
#2.)Spearman's test if we want to confirm correlation (pearson's assumes normally distributed vars)

# =============================================================================
# 2.) Write a function named plot_variable_pairs that accepts a dataframe
#     as input and plots all of the pairwise relationships along with the regression
#     line for each pair.
# ====
df.plot.scatter(
    x="annual_income",
    y="spending_score",
    title='Is there a relationship\nbetween annual income and spending score?',
    figsize=(8, 6),
)

# Conclusion:
# -not a linear relationship
# -looks like an "X"
# -looks like there might be clusters, the middle is very dense, the corners not so much  


df.head()


# Is there a relationship between age and spending score?

# Viz age by spending_score.
# Create age bins and compare

df.plot.scatter(y='spending_score', x='age', title='Is there a relationship between age and spending score?', figsize=(13, 8))


# Takeaways:

# spending score trails off for older individuals
# younger folks seem to have higher spending scores
# after age ~ 40, max(spending score) decreases



x = pd.Series(range(1, 11))
x


pd.cut(x, bins=[0, 7, 9, 11])

# 0     (0, 7]
# 1     (0, 7]
# 2     (0, 7]
# 3     (0, 7]
# 4     (0, 7]
# 5     (0, 7]
# 6     (0, 7]
# 7     (7, 9]
# 8     (7, 9]
# 9    (9, 11]
# dtype: category
# Categories (3, interval[int64]): [(0, 7] < (7, 9] < (9, 11]]

df['age_bin'] = pd.cut(df.age, [0, 40, 80])
df.head()






df.age_bin.value_counts().plot.barh(title='Distribution of age_bin')




#              gender  age  annual_income  spending_score   age_bin
# customer_id                                                      
# 64           Female   54             47              59  (40, 80]
# 49           Female   29             40              42   (0, 40]
# 25           Female   54             28              14  (40, 80]
# 137          Female   44             73               7  (40, 80]
# 177            Male   58             88              15  (40, 80]




sns.boxplot(data=df, y='spending_score', x='age_bin')



stats.levene(
    df[df.age <= 40].spending_score,
    df[df.age > 40].spending_score,
)

# LeveneResult(statistic=9.154464052447656, pvalue=0.0030460560375097914)

# =============================================================================
# Conclusion:
# 
# We reject the null hypothesis that the variance in spending score is the same for
#  folks less than or equal to 40 and above 40.
# 
# The variance is not equal.
# =============================================================================

stats.ttest_ind(
    df[df.age <= 40].spending_score,
    df[df.age > 40].spending_score,
    equal_var=False,
)
# Ttest_indResult(statistic=6.2431568169013785, pvalue=7.026347337563538e-09)



# Conclusion:

# We reject the null hypothesis that the average spending score for folks less
#  than or equal to 40 is the same as the average spending score for folks over 40.

df.head()

#              gender  age  annual_income  spending_score   age_bin
# customer_id                                                      
# 64           Female   54             47              59  (40, 80]
# 49           Female   29             40              42   (0, 40]
# 25           Female   54             28              14  (40, 80]
# 137          Female   44             73               7  (40, 80]
# 177            Male   58             88              15  (40, 80]

# If we control for age, does spending score differ across annual income?
# -Viz annual income by spending score for each age bin
# -Analyze each age bin separately




sns.relplot(data=df, y='spending_score', x='annual_income', col='age_bin')
# Takeaways:

# 0 through 40 group still has an x-shape
# 40+ crowd is just the bottom half of the x



# sns.relplot(
#     data=df,
#     y='spending_score',
#     x='annual_income',
#     col=pd.cut(df.age, bins=[0, 30, 40, 80]),
# )
# plt.suptitle("Do the different decades account\nfor the upper vs lower extremes?")
# plt.tight_layout()





# Takeaways:

# 30-40 almost looks linear
# 0-30 almost looks like a negative linear relationship
# 40-80 looks quadratic



df.head()


#Output:
    
#              gender  age  annual_income  spending_score   age_bin
# customer_id                                                      
# 64           Female   54             47              59  (40, 80]
# 49           Female   29             40              42   (0, 40]
# 25           Female   54             28              14  (40, 80]
# 137          Female   44             73               7  (40, 80]
# 177            Male   58             88              15  (40, 80]


sns.scatterplot(
    x='annual_income',
    y='spending_score',
    data=df[df.age <= 40],
    hue='gender',
)
plt.title("Does gender acccount for upper vs lower\nin the younger age group?")


# Aside: scatterplot vs relplot

# scatter plot works with axes
# relplot works with figures



# seaborn is built on top of matplotlib
# every matplotlib axis lives within a figure
# a figure can have 1 or more axes in it (2+ is when we have subplots)




sns.relplot(
    x='annual_income',
    y='spending_score',
    data=df,
    hue='gender',
    col='age_bin',
)
plt.title("Does gender acccount for upper vs lower\nin the younger age group?")\
    
    
#     Takeaways:

# gender isn't terribly informative in this context





df.head()


#              gender  age  annual_income  spending_score   age_bin
# customer_id                                                      
# 64           Female   54             47              59  (40, 80]
# 49           Female   29             40              42   (0, 40]
# 25           Female   54             28              14  (40, 80]
# 137          Female   44             73               7  (40, 80]
# 177            Male   58             88              15  (40, 80]



#  we control for annual income, does spending score differ across age?
# Because of the shape of annual income with spending score, I will create 3 bins of
#  income: [0, 40), [40, 70), [70, 140].




ax = df.annual_income.plot.hist()
ax.axvline(50, color='black')
ax.axvline(90, color='black')


df.head()


df['income_bin'] = pd.cut(df.annual_income, [0, 50, 90, 140])

plt.figure(figsize=(13, 7))
sns.scatterplot(
    x='age',
    y='spending_score',
    data=df,
    hue='income_bin',
)
plt.title("How does age compare to spending score\nwithin each income bin?")





df['income_bin'] = pd.cut(df.annual_income, [0, 40, 70, 140])

plt.figure(figsize=(13, 7))
sns.scatterplot(
    x='age',
    y='spending_score',
    data=df,
    hue='income_bin',
)
plt.title("How does age compare to spending score\nwithin each income bin?")




# =============================================================================
# Takeaways:
# 
# Summary
# annual income and spending score are good candidates for clustering
# older folks past some cutoff (40+) tend to not high values for spending score
# theres a good number of younger folks (30-) with low incomes and high spending scores
# gender didn't really seem to have an impact
# =============================================================================
# =============================================================================
# 3.) Write a function named months_to_years that accepts your telco churn
#     dataframe and returns a dataframe with a new feature tenure_years, in 
#     complete years as a customer.
# =============================================================================





# =============================================================================
# 4.) Write a function named plot_categorical_and_continuous_vars that accepts
#     \your dataframe and the name of the columns that hold the continuous and
#     categorical features and outputs 3 different plots for visualizing a
#     categorical variable and a continuous variable.
# =============================================================================

def get_distribution(df):
    for i in df:
        plt.title('{} Distribution'.format(i))
        plt.xlabel(i)
        plt.ylabel('count')
        df[i].hist(grid = False, bins = 100)
        plt.show()
        


# =============================================================================
# 5.) Save the functions you have written to create visualizations in your
#     
#    explore.py file. Rewrite your notebook code so that you are using the
#    functions imported from this file
# =============================================================================
def plot_variable_dist(df, figsize = (3,2)):
    '''
    This function is for exploring. Takes in a dataframe with variables you would like to 
    see the distribution of.
    Input the dataframe (either fully, or using .drop) with ONLY the columns you want to see plotted. 
    Optional argument figsize. Default it's small. 
    BTW if you just put list(df) it pulls out only the column names
    '''
    # loop through columns and use seaborn to plot distributions
    for col in list(df):
            plt.figure(figsize=figsize)
            plt.hist(data = df, x = col)
            plt.hist(df[np.isfinite(df[col])].values)
            plt.title(f'Distribution of {col}')
            plt.show()
            print(f'Number of Nulls: {df[col].isnull().sum()}')
            print('--------------------------------------')
def compare_to_target(df, target):
    for i in df:
        plt.figure(figsize=(8,3))
        plt.title('{} vs {}'.format(i,target))
        sns.scatterplot(data=df , y = target, x = i)
        plt.show()




# =============================================================================
# 6.) Explore your dataset with any other visualizations you think will be helpful.
# =============================================================================


def target_heat(df, target, method='pearson'):
    '''
    Use seaborn to create heatmap with coeffecient annotations to
    visualize correlation between all variables
    '''

    # define variable for corr matrix
    heat_churn = df.corr()[target][:-1]
    # set figure size
    fig, ax = plt.subplots(figsize=(30, 1))
    # define cmap for chosen color palette
    cmap = sns.diverging_palette(h_neg=220, h_pos=13, sep=25, as_cmap=True)
    # plot matrix turned to DataFrame
    sns.heatmap(heat_churn.to_frame().T, cmap=cmap, center=0,
                annot=True, fmt=".1g", cbar=False, square=True)
    #  improve readability of xticks, remove churn ytick
    plt.xticks(ha='right', va='top', rotation=35, rotation_mode='anchor')
    plt.yticks(ticks=[])
    # set title and print graphic
    plt.title(f'Correlation to {target}\n')
    plt.show()
    
    
    
def corr_test(data, x, y, alpha=0.05, r_type='pearson'):
    '''
    Performs a pearson or spearman correlation test and returns the r
    measurement as well as comparing the return p valued to the pass or
    default significance level, outputs whether to reject or fail to
    reject the null hypothesis
    
    '''
    
    # obtain r, p values
    if r_type == 'pearson':
        r, p = pearsonr(data[x], data[y])
    if r_type == 'spearman':
        r, p = spearmanr(data[x], data[y])
    # print reject/fail statement
    print(f'''{r_type:>10} r = {r:.2g}
+--------------------+''')
    if p < alpha:
        print(f'''
        Due to p-value {p:.2g} being less than our significance level of \
{alpha}, we may reject the null hypothesis 
        that there is not a linear correlation between "{x}" and "{y}."
        ''')
    else:
        print(f'''
        Due to p-value {p:.2g} being greater than our significance level of \
{alpha}, we fail to reject the null hypothesis 
        that there is not a linear correlation between "{x}" and "{y}."
        ''')
        
        
        
def plot_univariate(data, variable):
    '''
    This function takes the passed DataFrame the requested and plots a
    configured boxenplot and distrubtion for it side-by-side
    '''

    # set figure dimensions
    plt.figure(figsize=(30,8))
    # start subplot 1 for boxenplot
    plt.subplot(1, 2, 1)
    sns.boxenplot(x=variable, data=data)
    plt.axvline(data[variable].median(), color='pink')
    plt.axvline(data[variable].mean(), color='red')
    plt.xlabel('')
    plt.title('Enchanced Box Plot', fontsize=25)
    # start subplot 2 for displot
    plt.subplot(1, 2, 2)
    sns.histplot(data=data, x=variable, element='step', kde=True, color='cyan',
                                line_kws={'linestyle':'dashdot', 'alpha':1})
    plt.axvline(data[variable].median(), color='pink')
    plt.axvline(data[variable].mean(), color='red')
    plt.xlabel('')
    plt.ylabel('')
    plt.title('Distribution', fontsize=20)
    # set layout and show plot
    plt.suptitle(f'{variable} $[n = {data[variable].count():,}]$', fontsize=25)
    plt.tight_layout()
    plt.show()
        
def elbow_plot(df, col_list):
    '''
    Takes in a DataFrame and column list to use below method to find
    changes in inertia for increasing k in cluster creation methodology
    '''

    # set figure parameters
    plt.figure(figsize=(30, 15))
    # create series and apply increasing k values to test for inertia
    pd.Series({k: KMeans(k).fit(df[col_list])\
                            .inertia_ for k in range(2, 15)}).plot(marker='*')
    # define plot labels and visual components
    plt.xticks(range(2, 15))
    plt.xlabel('$k$')
    plt.ylabel('Inertia')
    plt.ylim(0,50000)
    plt.title('Changes in Inertia for Increasing $k$')
    plt.show()





def explore_clusters(df, col_list, k=2):
    '''
    Takes in a DataFrame, column list, and optional integer value for
    k to create clusters for the purpose of exploration, returns a
    DataFrame containing cluster group numbers and cluster centers
    '''

    # create kmeans object
    kmeans = KMeans(n_clusters=k, random_state=19)
    # fit kmeans
    kmeans.fit(df[col_list])
    # store predictions
    cluster_df = pd.DataFrame(kmeans.predict(df[col_list]), index=df.index,
                                                        columns=['cluster'])
    cluster_df = pd.concat((df[col_list], cluster_df), axis=1)
    # store centers
    center_df = cluster_df.groupby('cluster')[col_list].mean()
    
    return cluster_df, center_df, kmeans



def plot_clusters(cluster_df, center_df, x_var, y_var):
    '''
    Takes in cluster and centers DataFrame created by explore_clusters
    function and plots the passed x and y variables that make up that
    cluster group with different colors
    '''

    # define cluster_ column for better seaborn interpretation
    cluster_df['cluster_'] = 'cluster_' + cluster_df.cluster.astype(str)
    # set scatterplot and dimensions
    plt.figure(figsize=(28, 14))
    sns.scatterplot(x=x_var, y=y_var, data=cluster_df, hue='cluster_', s=100)
    # plot cluster centers
    center_df.plot.scatter(x=x_var, y=y_var, ax=plt.gca(), s=300, c='k',
                                        edgecolor='w', marker='$\\bar{x}$')
    # set labels and legend, show
    plt.xlabel(f'\n{x_var}\n', fontsize=20)
    plt.ylabel(f'\n{y_var}\n', fontsize=20)
    plt.title('\nClusters and Their Centers\n', fontsize=30)
    plt.legend(bbox_to_anchor=(0.95,0.95), fontsize=20)

    plt.show() 



def plot_three_d_clusters(cluster_df, center_df, x_var, y_var, z_var):
    '''
    Takes in cluster and centers DataFrame created by explore_clusters
    function and creates a three dimesnional plot of the passed x, y,
    and z variables that make up that cluster group with different
    colors
    '''

    # set figure and axes
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111, projection='3d')    
    # set clusters for each cluster passed in arguments
    # set x, y, z for cluster 0
    x0 = cluster_df[cluster_df['cluster'] == 0][x_var]
    y0 = cluster_df[cluster_df['cluster'] == 0][y_var]
    z0 = cluster_df[cluster_df['cluster'] == 0][z_var]
    # set x, y, z for cluster 1
    x1 = cluster_df[cluster_df['cluster'] == 1][x_var]
    y1 = cluster_df[cluster_df['cluster'] == 1][y_var]
    z1 = cluster_df[cluster_df['cluster'] == 1][z_var]
    # set x, y, z for each additional cluster
    if len(center_df) > 2:
        x2 = cluster_df[cluster_df['cluster'] == 2][x_var]
        y2 = cluster_df[cluster_df['cluster'] == 2][y_var]
        z2 = cluster_df[cluster_df['cluster'] == 2][z_var]
    if len(center_df) > 3:
        x3 = cluster_df[cluster_df['cluster'] == 3][x_var]
        y3 = cluster_df[cluster_df['cluster'] == 3][y_var]
        z3 = cluster_df[cluster_df['cluster'] == 3][z_var]
    if len(center_df) > 4:
        x4 = cluster_df[cluster_df['cluster'] == 4][x_var]
        y4 = cluster_df[cluster_df['cluster'] == 4][y_var]
        z4 = cluster_df[cluster_df['cluster'] == 4][z_var]
    if len(center_df) > 5:
        x5 = cluster_df[cluster_df['cluster'] == 5][x_var]
        y5 = cluster_df[cluster_df['cluster'] == 5][y_var]
        z5 = cluster_df[cluster_df['cluster'] == 5][z_var]
        
    # set centers for each cluster passed in arguments
    # set centers for clusters 0, 1
    zero_center = center_df[center_df.index == 0]
    one_center = center_df[center_df.index == 1]
    # set centers for each additional clusters
    if len(center_df) > 2:
        two_center = center_df[center_df.index == 2]
    if len(center_df) > 3:
        three_center = center_df[center_df.index == 3]
    if len(center_df) > 4:
        four_center = center_df[center_df.index == 4]
    if len(center_df) > 5:
        five_center = center_df[center_df.index == 5]
    if len(center_df) > 6:
        six_center = center_df[center_df.index == 6]
        
    # plot clusters and their centers for each cluster passed in arguments
    # plot cluster 0 with center
    ax.scatter(x0, y0, z0, s=100, c='c', edgecolor='k', marker='o',
                                                    label='Cluster 0')
    ax.scatter(zero_center[x_var], zero_center[y_var], zero_center[z_var],
                                    s=300, c='c', marker='$\\bar{x}$')
    # plot cluster 1 with center
    ax.scatter(x1, y1, z1, s=100, c='y', edgecolor='k', marker='o',
                                                    label='Cluster 1')
    ax.scatter(one_center[x_var], one_center[y_var], one_center[z_var],
                                    s=300, c='y', marker='$\\bar{x}$')
    # plot each additional cluster passed in arguments
    if len(center_df) > 2:
        ax.scatter(x2, y2, z2, s=100, c='m', edgecolor='k', marker='o',
                                                    label='Cluster 2')
        ax.scatter(two_center[x_var], two_center[y_var], two_center[z_var],
                                    s=300, c='m', marker='$\\bar{x}$')
    if len(center_df) > 3:
        ax.scatter(x3, y3, z3, s=100, c='k', edgecolor='w', marker='o',
                                                    label='Cluster 3')
        ax.scatter(three_center[x_var],three_center[y_var],three_center[z_var],
                                    s=300, c='k', marker='$\\bar{x}$')
    if len(center_df) > 4:
        ax.scatter(x4, y4, z4, s=100, c='r', edgecolor='k', marker='o',
                                                    label='Cluster 4')
        ax.scatter(four_center[x_var], four_center[y_var], four_center[z_var],
                                    s=300, c='r', marker='$\\bar{x}$')
    if len(center_df) > 5:
        ax.scatter(x5, y5, z5, s=100, c='g', edgecolor='k', marker='o',
                                                    label='Cluster 5')
        ax.scatter(five_center[x_var], five_center[y_var], five_center[z_var],
                                    s=300, c='g', marker='$\\bar{x}$')
    # if len(center_df) > 6:
    #     ax.scatter(x6, y6, z6, s=100, c='b', edgecolor='k', marker='o',
    #                                                 label='Cluster 6')
        ax.scatter(six_center[x_var], six_center[y_var], six_center[z_var],
                                    s=300, c='b', marker='$\\bar{x}$')
        
    # set labels, title, and legend
    ax.set_xlabel(f'\n$x =$ {x_var}', fontsize=15)
    ax.set_ylabel(f'\n$y =$ {y_var}', fontsize=15)
    ax.set_zlabel(f'\n$z =$ {z_var}', fontsize=15)
    plt.title('Clusters and Their Centers', fontsize=30)
    plt.legend(bbox_to_anchor=(0.975,0.975), fontsize=15)

    plt.show()





# =============================================================================
# 7.) In a seperate notebook, use the functions you have developed in this exercise
#     with the mall_customers dataset in the Codeup database server. You will need
#     to write a sql query to acquire your data. Make spending_score your target variable.
# =============================================================================










# =============================================================================
#                     Exercises II - Challenge
# =============================================================================


# =============================================================================
# Our Zillow scenario continues:
# 
# As a Codeup data science graduate, you want to show off your skills to
#  the Zillow data science team in hopes of getting an interview for a position
#  you saw pop up on LinkedIn. You thought it might look impressive to build an
#  end-to-end project in which you use some of their Kaggle data to predict
#  property values using some of their available features; who knows, you might
#  even do some feature engineering to blow them away. Your goal is to predict
#  the values of single unit properties using the observations from 2017.
# 
# In these exercises, you will run through the stages of exploration as you
#  continue to work toward the above goal.
# =============================================================================




# =============================================================================
# 1.) Use the functions you created above to explore your Zillow train
#     dataset in your explore.ipynb notebooK
# =============================================================================






# =============================================================================
# 2.) Come up with some initial hypotheses based on your goal of predicting
#     property value.
# =============================================================================




# =============================================================================
# 3.)Visualize all combinations of variables in some way.
# =============================================================================





# =============================================================================
# 4.)Run the appropriate statistical tests where needed.
# =============================================================================



# =============================================================================
# 5.) What independent variables are correlated with the dependent variable, home value?
# =============================================================================



# =============================================================================
# 6.) Which independent variables are correlated with other independent
#     variables (bedrooms, bathrooms, year built, square feet)?
# =============================================================================





# =============================================================================
# 7.) Make sure to document your takeaways from visualizations and statistical
#     tests as well as the decisions you make throughout your procesS.
# =============================================================================
