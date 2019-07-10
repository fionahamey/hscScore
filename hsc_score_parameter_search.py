#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import scanpy as sc

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from sklearn.decomposition import PCA

sc.logging.print_version_and_date()

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# Load the data
data_dir = './Data/'
fig_dir = './Figdir/'

# Define output file
output_file = 'model_selection_cv_scores.txt'

# Counts data
wilson_counts = sc.read(data_dir + 'wilson_HTSEQ_results_edit.txt').transpose()

# Gene table
gene_table = pd.read_csv(data_dir + 'wilson_ensembl_gene_table_81.txt',
                         index_col=1, sep='\t')

# Filter to protein coding and not mitochondrial RNA
protein_coding_genes = gene_table[gene_table['Gene type'] == 'protein_coding']
protein_coding_nuclear = list(protein_coding_genes[
    [gene[0:3] != 'mt-' for gene in protein_coding_genes.index]].index)

wilson_counts = sc.AnnData(
        wilson_counts[:, np.array([gene in protein_coding_nuclear for
                                   gene in wilson_counts.var_names])])

# Add MolO score information
molo_scores = pd.read_csv(data_dir + 'wilson_rna_seq_molo_scores.csv',
                          index_col=1)
molo_scores.index = [cell.replace("SLX.", "SLX-") for
                     cell in molo_scores.index]

# Filter to only cells that have been MolO scored
molo_scored_cells = molo_scores.index
wilson_scored = [cell in molo_scored_cells for cell in wilson_counts.obs_names]
wilson_counts = wilson_counts[np.array(wilson_scored), :]

# Add info to count data
wilson_counts.obs['molo_score'] = molo_scores[
        'molo_score'].loc[wilson_counts.obs_names]
wilson_counts.obs['PC1'] = molo_scores['PC1'].loc[wilson_counts.obs_names]
wilson_counts.obs['PC2'] = molo_scores['PC2'].loc[wilson_counts.obs_names]

# Normalise data
wilson_adata = wilson_counts.copy()
sc.pp.normalize_total(wilson_adata)
sc.pp.log1p(wilson_adata)

# Find variable genes
sc.pp.highly_variable_genes(wilson_adata)
highly_variable_gene_list = list(
        wilson_adata.var_names[wilson_adata.var['highly_variable']])

# Read in MolO genes
molo_genes = pd.read_csv(data_dir + 'wilson_nomo_molo_genes.csv')
print(molo_genes.head())
molo_gene_list = [gene for gene in wilson_adata.var_names if gene in list(
        molo_genes['Gene'])]


# Data normalisation methods
# Rank normalisation function
def rank_normalise_genes(adata):

    exprs = pd.DataFrame(adata.X)
    ranked = exprs.rank(axis=1, method='average')
    ranked = np.array(ranked)

    adata_ranked = adata.copy()
    adata_ranked.X = ranked

    return(adata_ranked)


# Total counts normalisation function
def total_count_normalised(adata, target_sum):

    adata_total_count = adata.copy()
    sc.pp.normalize_total(adata_total_count, target_sum=target_sum)
    sc.pp.log1p(adata_total_count)

    return(adata_total_count)


# All genes
wilson_adata_rank = wilson_counts.copy()
wilson_adata_rank = rank_normalise_genes(wilson_adata_rank)

wilson_adata_total_count = wilson_counts.copy()
target_sum_all = np.median(np.sum(wilson_adata_total_count.X, axis=1))
wilson_adata_total_count = total_count_normalised(wilson_adata_total_count, target_sum_all)


# Variable genes
wilson_adata_var_rank = wilson_counts.copy()
wilson_adata_var_rank = sc.AnnData(
        wilson_adata_var_rank[:, highly_variable_gene_list])
wilson_adata_var_rank = rank_normalise_genes(wilson_adata_var_rank)

wilson_adata_var_total_count = wilson_counts.copy()
wilson_adata_var_total_count = sc.AnnData(
        wilson_adata_var_total_count[:, highly_variable_gene_list])
target_sum_var = np.median(np.sum(wilson_adata_var_total_count.X, axis=1))
wilson_adata_var_total_count = total_count_normalised(wilson_adata_var_total_count, target_sum_var)

wilson_adata_molo_rank = wilson_counts.copy()
wilson_adata_molo_rank = sc.AnnData(wilson_adata_molo_rank[:, molo_gene_list])
wilson_adata_molo_rank = rank_normalise_genes(wilson_adata_molo_rank)

wilson_adata_molo_total_count = wilson_counts.copy()
wilson_adata_molo_total_count = sc.AnnData(wilson_adata_molo_total_count[:, molo_gene_list])
target_sum_molo = np.median(np.sum(wilson_adata_molo_total_count.X, axis=1))
wilson_adata_molo_total_count = total_count_normalised(wilson_adata_molo_total_count, target_sum_molo)


# Regression function
def testRegression(regr, X, y, param_grid, random_state=7, test_size=0.25,
                   cv=5, plt_title='', save_plt_title=None, pca=False):

    """ Test sklearn regression function

    Implementing a test pipeline for a sklearn regression function on a dataset
    Returns various plots and metrics of performance

    Parameters
    -----------
    regr : sklearn regression function
    X : data variables
    y : data observations we are trying to predict
    param_grid : parameters to search in GridSearchCV function
    random_state : for reproducible results
    test_size : fraction of observations in test size for test-training split
    cv : number of folds in cross validation
    """

    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)

    # Make pipeline for scaling followed by the regression
    if pca:
        pipe = Pipeline(steps=[('scale', StandardScaler()),
                               ('pca', PCA()),
                               ('regr', regr)])
    else:
        pipe = Pipeline(steps=[('scale', StandardScaler()),
                               ('regr', regr)])

    # Perform search for pararmeters
    regr_search = GridSearchCV(pipe, param_grid, cv=cv, iid=True)
    regr_search.fit(X_train, y_train)

    # Best parameters
    best_parameters = regr_search.best_params_

    means = regr_search.cv_results_['mean_test_score']
    stds = regr_search.cv_results_['std_test_score']
    params = np.array(regr_search.cv_results_['params'])

    best_index = np.where(params == best_parameters)
    best_parameters_mean = means[best_index]
    best_parameters_std = stds[best_index]

    cv_0_score = regr_search.cv_results_['split0_test_score'][best_index]
    cv_1_score = regr_search.cv_results_['split1_test_score'][best_index]
    cv_2_score = regr_search.cv_results_['split2_test_score'][best_index]
    cv_3_score = regr_search.cv_results_['split3_test_score'][best_index]
    cv_4_score = regr_search.cv_results_['split4_test_score'][best_index]

    # Assess results of prediction
    y_true, y_pred = y_test, regr_search.predict(X_test)
    score = regr_search.score(X_test, y_test)

    fig, ax = pl.subplots()
    ax.scatter(y_true, y_pred, s=25, zorder=10)

    lims = [0, np.max([ax.get_xlim(), ax.get_ylim()])]

    # now plot both limits against eachother
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    pl.title("%s Score: %.2f" % (plt_title, score))
    pl.xlabel('Real MolO scores')
    pl.ylabel('Predicted MolO scores')
    if save_plt_title:
        pl.savefig(fig_dir + save_plt_title + '.pdf')

    return({'regr_search': regr_search,
            'best_parameters': best_parameters,
            'mean_score': best_parameters_mean[0],
            'sd_score': best_parameters_std[0],
            'cv_0_score': cv_0_score[0],
            'cv_1_score': cv_1_score[0],
            'cv_2_score': cv_2_score[0],
            'cv_3_score': cv_3_score[0],
            'cv_4_score': cv_4_score[0],
            'test_score': score})


def test_regression_different_genes_norms(regr, param_grid,
                                          regression_type, pca=False):

    if pca:
        pca_status = 'Yes'
    else:
        pca_status = 'No'

    # All genes ranked
    print()
    print('----------------------------------------------------')
    print('All genes ranking norm')
    regression = testRegression(regr=regr,
                                X=wilson_adata_rank.X,
                                y=wilson_adata_rank.obs['molo_score'],
                                param_grid=param_grid,
                                plt_title='All genes ranking:',
                                save_plt_title=regression_type +
                                '_all_genes_ranking_norm',
                                pca=pca)

    print('Best parameters found:')
    print(regression['best_parameters'])
    print()
    print("%0.3f (+/-%0.03f) for best parameters" % (regression['mean_score'],
                                                     regression[
                                                             'sd_score'] * 2))

    result_list = ['All',
                   'Ranking',
                   pca_status,
                   regression_type,
                   regression['best_parameters'],
                   regression['cv_0_score'],
                   regression['cv_1_score'],
                   regression['cv_2_score'],
                   regression['cv_3_score'],
                   regression['cv_4_score'],
                   regression['mean_score'],
                   regression['sd_score'],
                   regression['test_score']]
    result_list = [str(r) for r in result_list]

    with open(output_file, 'a+') as f:
        f.write('\t'.join(result_list) + '\n')

    # All genes total_count
    print()
    print('----------------------------------------------------')
    print('All genes total_count norm')
    regression = testRegression(regr=regr,
                                X=wilson_adata_total_count.X,
                                y=wilson_adata_total_count.obs['molo_score'],
                                param_grid=param_grid,
                                plt_title='All genes total_count:',
                                save_plt_title=regression_type +
                                '_all_genes_total_count_norm',
                                pca=pca)

    print('Best parameters found:')
    print(regression['best_parameters'])
    print()
    print("%0.3f (+/-%0.03f) for best parameters" % (regression['mean_score'],
                                                     regression[
                                                             'sd_score'] * 2))

    result_list = ['All',
                   'Total_count',
                   pca_status,
                   regression_type,
                   regression['best_parameters'],
                   regression['cv_0_score'],
                   regression['cv_1_score'],
                   regression['cv_2_score'],
                   regression['cv_3_score'],
                   regression['cv_4_score'],
                   regression['mean_score'],
                   regression['sd_score'],
                   regression['test_score']]
    result_list = [str(r) for r in result_list]

    with open(output_file, 'a+') as f:
        f.write('\t'.join(result_list) + '\n')

    # Variable genes ranked
    print()
    print('----------------------------------------------------')
    print('Variable genes ranking norm')
    regression = testRegression(regr=regr,
                                X=wilson_adata_var_rank.X,
                                y=wilson_adata_var_rank.obs['molo_score'],
                                param_grid=param_grid,
                                plt_title='Var genes ranking:',
                                save_plt_title=regression_type +
                                '_var_genes_ranking_norm',
                                pca=pca)

    print('Best parameters found:')
    print(regression['best_parameters'])
    print()
    print("%0.3f (+/-%0.03f) for best parameters" % (regression['mean_score'],
                                                     regression[
                                                             'sd_score'] * 2))

    result_list = ['Variable',
                   'Ranking',
                   pca_status,
                   regression_type,
                   regression['best_parameters'],
                   regression['cv_0_score'],
                   regression['cv_1_score'],
                   regression['cv_2_score'],
                   regression['cv_3_score'],
                   regression['cv_4_score'],
                   regression['mean_score'],
                   regression['sd_score'],
                   regression['test_score']]
    result_list = [str(r) for r in result_list]

    with open(output_file, 'a+') as f:
        f.write('\t'.join(result_list) + '\n')

    # Variable genes total_count
    print()
    print('----------------------------------------------------')
    print('Variable genes total_count norm')
    regression = testRegression(regr=regr,
                                X=wilson_adata_var_total_count.X,
                                y=wilson_adata_var_total_count.obs['molo_score'],
                                param_grid=param_grid,
                                plt_title='Var genes total_count:',
                                save_plt_title=regression_type +
                                '_var_genes_total_count_norm',
                                pca=pca)

    print('Best parameters found:')
    print(regression['best_parameters'])
    print()
    print("%0.3f (+/-%0.03f) for best parameters" % (regression['mean_score'],
                                                     regression[
                                                             'sd_score'] * 2))

    result_list = ['Variable',
                   'Total_count',
                   pca_status,
                   regression_type,
                   regression['best_parameters'],
                   regression['cv_0_score'],
                   regression['cv_1_score'],
                   regression['cv_2_score'],
                   regression['cv_3_score'],
                   regression['cv_4_score'],
                   regression['mean_score'],
                   regression['sd_score'],
                   regression['test_score']]
    result_list = [str(r) for r in result_list]

    with open(output_file, 'a+') as f:
        f.write('\t'.join(result_list) + '\n')

    # Molo genes ranking
    print()
    print('----------------------------------------------------')
    print('Molo genes ranking norm')
    regression = testRegression(regr=regr,
                                X=wilson_adata_molo_rank.X,
                                y=wilson_adata_molo_rank.obs['molo_score'],
                                param_grid=param_grid,
                                plt_title='Molo genes ranking:',
                                save_plt_title=regression_type +
                                '_molo_genes_ranking_norm',
                                pca=pca)

    print('Best parameters found:')
    print(regression['best_parameters'])
    print()
    print("%0.3f (+/-%0.03f) for best parameters" % (regression['mean_score'],
                                                     regression[
                                                             'sd_score'] * 2))

    result_list = ['Molo',
                   'Ranking',
                   pca_status,
                   regression_type,
                   regression['best_parameters'],
                   regression['cv_0_score'],
                   regression['cv_1_score'],
                   regression['cv_2_score'],
                   regression['cv_3_score'],
                   regression['cv_4_score'],
                   regression['mean_score'],
                   regression['sd_score'],
                   regression['test_score']]
    result_list = [str(r) for r in result_list]

    with open(output_file, 'a+') as f:
        f.write('\t'.join(result_list) + '\n')

    # Molo genes total_count
    print()
    print('----------------------------------------------------')
    print('Molo genes total_count norm')
    regression = testRegression(regr=regr,
                                X=wilson_adata_molo_total_count.X,
                                y=wilson_adata_molo_total_count.obs['molo_score'],
                                param_grid=param_grid,
                                plt_title='Molo genes total_count:',
                                save_plt_title=regression_type +
                                '_molo_genes_total_count_norm',
                                pca=pca)

    print('Best parameters found:')
    print(regression['best_parameters'])
    print()
    print("%0.3f (+/-%0.03f) for best parameters" % (regression['mean_score'],
                                                     regression[
                                                             'sd_score'] * 2))

    result_list = ['Molo',
                   'Total_count',
                   pca_status,
                   regression_type,
                   regression['best_parameters'],
                   regression['cv_0_score'],
                   regression['cv_1_score'],
                   regression['cv_2_score'],
                   regression['cv_3_score'],
                   regression['cv_4_score'],
                   regression['mean_score'],
                   regression['sd_score'],
                   regression['test_score']]
    result_list = [str(r) for r in result_list]

    with open(output_file, 'a+') as f:
        f.write('\t'.join(result_list) + '\n')


# ---------------------------------------------
# Specify output file for saving results
with open(output_file, 'w+') as f:
    f.write('\t'.join(['Gene_set', 'Normalisation', 'PCA', 'Model',
                       'Best_parameters', 'cv_1_score', 'cv_2_score',
                       'cv_3_score', 'cv_4_score', 'cv_5_score',
                       'Average_score', 'SD', 'Score_on_test_data\n']))


# ---------------------------------------------
# Iterate through different regression models

# Random forest regression
print('########################################')
print('Random forest regression')
test_regression_different_genes_norms(
        regr=RandomForestRegressor(random_state=7, n_estimators=100),
        param_grid=[{'regr__max_depth': [2, 5, 10, 15, None],
                     'regr__min_samples_leaf': [1, 2, 5, 10],
                     'regr__min_samples_split': [2, 5, 10],
                     'regr__max_features': [5, 'auto', 'sqrt', 'log2']}],
        regression_type='random_forest',
        pca=False)

test_regression_different_genes_norms(
        regr=RandomForestRegressor(random_state=7, n_estimators=100),
        param_grid=[{'regr__max_depth': [2, 5, 10, 15, None],
                     'regr__min_samples_leaf': [1, 2, 5, 10],
                     'regr__min_samples_split': [2, 5, 10],
                     'regr__max_features': [5, 'auto', 'sqrt', 'log2'],
                     'pca__n_components': [5, 20, 50]}],
        regression_type='random_forest_pca',
        pca=True)

# Linear regression
print('########################################')
print('Linear regression')
test_regression_different_genes_norms(
        regr=LinearRegression(),
        param_grid=[{'regr__fit_intercept': ['True', 'False']}],
        regression_type='linear_regression',
        pca=False)

test_regression_different_genes_norms(
        regr=LinearRegression(),
        param_grid=[{'regr__fit_intercept': ['True', 'False'],
                     'pca__n_components': [5, 20, 50]}],
        regression_type='linear_regression_pca',
        pca=True)

# Support vector machines
print('########################################')
print('Support vector machines')
test_regression_different_genes_norms(
        regr=svm.SVR(),
        param_grid=[{'regr__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                     'regr__C': [0.1, 1.0, 10],
                     'regr__gamma': [0.01, 0.1, 1.0, 10],
                     'regr__degree': [2, 3, 4, 5],
                     'regr__coef0': [0, -1, 1],
                     'regr__epsilon': [0.1, 0.01, 1]}],
        regression_type='svm',
        pca=False)

test_regression_different_genes_norms(
        regr=svm.SVR(),
        param_grid=[{'regr__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                     'regr__C': [0.1, 1.0],
                     'regr__gamma': [0.01, 0.1, 1.0, 10],
                     'regr__degree': [2, 3],
                     'regr__coef0': [0, -1, 1],
                     'regr__epsilon': [0.1, 0.01],
                     'pca__n_components': [20, 50]}],
        regression_type='svm_pca',
        pca=True)

# Nearest neighbour regession
print('########################################')
print('Nearest neighbour regression')
test_regression_different_genes_norms(
        regr=KNeighborsRegressor(),
        param_grid=[{'regr__p': [1, 2],
                     'regr__n_neighbors': [3, 5, 10, 15]}],
        regression_type='knn',
        pca=False)

test_regression_different_genes_norms(
        regr=KNeighborsRegressor(),
        param_grid=[{'regr__p': [1, 2],
                     'regr__n_neighbors': [3, 5, 10, 15],
                     'pca__n_components': [5, 20, 50]}],
        regression_type='knn_pca',
        pca=True)

# Neural network models
print('#######################################')
print('MLP regressor neural net')
test_regression_different_genes_norms(
        regr=MLPRegressor(random_state=7, max_iter=500),
        param_grid=[{'regr__activation': ['identity', 'logistic',
                                          'tanh', 'relu'],
                     'regr__hidden_layer_sizes': [(100, 50, 100), (100, 100),
                                                  (100, 100, 100), (50, 50),
                                                  (50, 50, 50), (50, 25, 50)],
                     'regr__alpha': 10.0 ** -np.arange(1, 5),
                     'regr__solver': ['lbfgs', 'adam']}],
        regression_type='mlp',
        pca=False)

test_regression_different_genes_norms(
        regr=MLPRegressor(random_state=7, max_iter=500),
        param_grid=[{'regr__activation': ['identity', 'logistic',
                                          'tanh', 'relu'],
                     'regr__hidden_layer_sizes': [(100, 50, 100), (100, 100),
                                                  (100, 100, 100), (50, 50),
                                                  (50, 50, 50), (50, 25, 50)],
                     'regr__alpha': 10.0 ** -np.arange(1, 5),
                     'regr__solver': ['lbfgs', 'adam'],
                     'pca__n_components': [5, 20, 50]}],
        regression_type='mlp_pca',
        pca=True)
