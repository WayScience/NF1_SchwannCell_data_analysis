#!/usr/bin/env python
# coding: utf-8

# # Perform a t-test between well correlations of the same or different genotypes

# ## Import libraries

# In[1]:


from scipy.stats import ttest_ind
import pathlib
import pandas as pd


# ## Load in correlation data

# In[2]:


# Path to correlation per plate results
corr_results_dir = pathlib.Path("../../0.data_analysis/plate_correlation_analyses/construct_correlation_data")

# Load data
corr_results_file = corr_results_dir / "well_agg_plate_genotype_correlations.parquet"
corr_results_df = pd.read_parquet(corr_results_file)

# Add a new column `same_genotype` to check if the correlation row is comparing between the same genotype
corr_results_df['same_genotype'] = corr_results_df['Metadata_genotype__group0'] == corr_results_df['Metadata_genotype__group1']

# Add a new column `same_plate` to check if the correlation row is comparing between the same plate
corr_results_df['same_plate'] = corr_results_df['Metadata_Plate__group0'] == corr_results_df['Metadata_Plate__group1']

# Display dimensions and first few rows of the DataFrame
print(corr_results_df.shape)
corr_results_df.head()


# ## Perform two sample t-test

# In[3]:


# Split the DataFrame based on the `same_genotype` column
same_genotype_df = corr_results_df[corr_results_df['same_genotype'] == True]
different_genotype_df = corr_results_df[corr_results_df['same_genotype'] == False]

# Perform a t-test between the two groups
# Replace 'your_column_of_interest' with the column you want to test
t_stat, p_value = ttest_ind(same_genotype_df['correlation'], 
                            different_genotype_df['correlation'])

print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")


# A large t-statistic and very low p-value indicates we can reject the null hypothesis and conclude that overall single-cell populations at the well level that are from the same genotype have a significantly different mean than the wells with different genotypes.

# ## Show the means of the different distributions

# In[4]:


same_genotype_mean = same_genotype_df['correlation'].mean()
different_genotype_mean = different_genotype_df['correlation'].mean()

print(f"Mean (same_genotype): {same_genotype_mean}")
print(f"Mean (different_genotype): {different_genotype_mean}")

