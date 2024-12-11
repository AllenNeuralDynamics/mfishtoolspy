import numpy as np
import pandas as pd
import re
from scipy.stats import rankdata
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist, pdist, squareform
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from dask import delayed, compute
from dask.distributed import Client
import dask


def build_mapping_based_marker_panel_testing(map_data, mapping_median_data=None, mapping_call=None, 
                                    mapping_to_group=None, group_median_data=None, num_iter_each_addition=100,
                                    panel_size=50, num_subsample=50, na_str='None', use_parallel=True,
                                    max_fc_gene=1000, qmin=0.75, seed=None, current_panel=None, current_metric=None,
                                    panel_min=5, verbose=True, corr_mapping=True, 
                                    optimize="correlation_distance", group_distance=None, 
                                    cluster_genes=None, dend=None, percent_gene_subset=100):
    """
    Builds a panel of genes based on mapping to a reference data set.

    #CAUTION: This function is recursive, so the inputs should be a copy of the original data.

    Parameters:
    map_data (pd.DataFrame): Data to be mapped.
    mapping_median_data (pd.DataFrame): Precomputed medians for clustering mapping. If None, it is computed.
    mapping_call (pd.Series): Mapping assignment of the columns in map_data.
    mapping_to_group (pd.DataFrame): Grouping assignment of the columns in map_data. If None, all the mappings will be used.
    num_iter_each_addition (int): Number of iterations to add each gene. If num_subsample is None, this is ignored.
    panel_size (int): Number of genes to include in the panel.
    num_subsample (int): Number of cells to subsample from each group.
    na_str (str): String to replace NA values with. Used for correlation-based mapping (group_distance, mapping_to_group, def get_top_match).
    use_parallel (bool): Whether to use parallel processing for each iteration within one gene addition.
    max_fc_gene (int): Maximum number of genes to use for filtering.
    qmin (float): Quantile to use for filtering.
    seed (int): Random seed for reproducibility.
    current_panel (list): List of genes to start with.
    panel_min (int): Minimum number of genes to start with.
    verbose (bool): Whether to print progress messages.
    corr_mapping (bool): Whether to use correlation-based mapping.
    optimize (str): Optimization criterion for gene selection.
    group_distance (np.ndarray): Pairwise group distances.
    cluster_genes (list): List of genes to use for cluster distance calculation.
    dend (np.ndarray): Dendrogram structure for cluster distance calculation.
    percent_gene_subset (float): Percentage of genes to consider for mapping.

    Returns:
    list: List of genes in the panel.
    list: List of metrics for each gene addition.
    """

    if optimize == "dendrogram_height" and dend is None:
        return "Error: dendrogram not provided"
    
    if mapping_median_data is None:
        mapping_median_data = pd.DataFrame()
        if mapping_call is None:
            raise ValueError("Both mapping_call and mapping_median_data must be provided if mapping_median_data is not provided.")
        if type(mapping_call) is not pd.Series:
            mapping_call = pd.Series(mapping_call, index=map_data.columns)
        mapping_median_data = map_data.groupby(mapping_call, axis=1).median()
    
    if mapping_median_data.index.isnull().any():
        mapping_median_data.index = map_data.index
    
    if mapping_to_group is None: # self mapping (i.e., mapping = group)
        mapping_to_group = pd.Series(mapping_call.values, index=mapping_call.values).drop_duplicates()
    if na_str not in mapping_to_group.index.values:
        mapping_to_group[na_str] = na_str # to handle missing values after get_top_match
    group_call = mapping_call.map(mapping_to_group)
    
    if group_median_data is None:
        group_median_data = pd.DataFrame()
        group_median_data = map_data.groupby(group_call, axis=1).median()
    
    if optimize == "fraction_correct":
        group_distance = None
    elif optimize == "correlation_distance":
        if group_distance is None:
            cor_dist = lambda x: 1 - np.corrcoef(x)
            if cluster_genes is None:
                cluster_genes = group_median_data.index
            cluster_genes = list(set(cluster_genes).intersection(set(group_median_data.index)))
            group_distance = pd.DataFrame(cor_dist(group_median_data.loc[cluster_genes, :].T),
                                            index=group_median_data.columns, columns=group_median_data.columns)
        # assign na_str values to the group_distance with maximum group_distance value
        max_group_distance = group_distance.values.max()
        if na_str not in group_distance.columns:
            group_distance[na_str] = max_group_distance
    else:
        raise ValueError("Invalid optimization criterion. Please choose 'fraction_correct' or 'correlation_distance'.")    
    # if optimize == "dendrogram_height":
    #     # Custom make_LCA_table and get_node_height functions need to be implemented
    #     lca_table = make_LCA_table(dend)
    #     group_distance = 1 - get_node_height(dend)[lca_table]
    #     optimize = "correlation_distance"

    # Calculate the gene expression difference between the 100th percentile cluster and the qmin percentile cluster
    # To be used for filtering (if not filtereed before)
    rank_expr_diff = rankdata(mapping_median_data.apply(lambda x: np.diff(np.percentile(x, [100 * qmin, 100])), axis=1))
    
    if mapping_median_data.shape[0] > max_fc_gene: # filter based on rank_expr_diff
        keep_genes = np.array(mapping_median_data.index)[rank_expr_diff <= max_fc_gene] # rankdata rank starts at 1
        map_data = map_data.loc[keep_genes, :]
        mapping_median_data = mapping_median_data.loc[keep_genes, :]

    panel_min = max(2, panel_min)
    if current_panel is None or len(current_panel) < panel_min:
        panel_min = max(2, panel_min - (len(current_panel) if current_panel else 0))
        current_panel = list(set(current_panel or []).union(set(mapping_median_data.index[rank_expr_diff.argsort()[:panel_min]])))
        if verbose:
            print(f"Setting starting panel as: {', '.join(current_panel)}")
    if current_metric is None:
        current_metric = []
    
    if len(current_panel) < panel_size:
        other_genes = list(set(map_data.index).difference(set(current_panel)))
        if percent_gene_subset < 100:
            if seed is not None:
                np.random.seed(seed + len(current_panel))
            other_genes = np.random.choice(other_genes, size=int(len(other_genes) * percent_gene_subset / 100), replace=False)

        if num_subsample is None: # use all samples, no iteration
            num_iter_each_addition = 1

            
        match_count = np.zeros((len(other_genes), num_iter_each_addition))
        # group_labels = group_median_data.columns # for flattening group_distance, just in case
        
        
        if use_parallel:
            client = Client()
            task = []
            if seed is None:
                tasks = [delayed(_run_one_iter_parallel)(map_data, group_call, mapping_call, num_subsample, 
                                                         seed, mapping_median_data, other_genes, current_panel, 
                                                         group_distance, mapping_to_group, corr_mapping, 
                                                         na_str=na_str) for iter_num in range(num_iter_each_addition)]
            else:
                tasks = [delayed(_run_one_iter_parallel)(map_data, group_call, mapping_call, num_subsample, 
                                                         seed_iter, mapping_median_data, other_genes, current_panel, 
                                                         group_distance, mapping_to_group, corr_mapping, 
                                                         na_str=na_str) for seed_iter in range(seed, seed + num_iter_each_addition)]
                seed += num_iter_each_addition
            results = compute(*tasks, num_workers=dask.system.cpu_count()-1)
            client.close()

            match_count = np.stack(results)

        else:
            for iter_num in range(num_iter_each_addition):
                if num_subsample is not None:
                    keep_sample = subsample_cells(group_call, num_subsample, seed) # subsample from group_call
                    map_data_iter = map_data.loc[:, keep_sample]
                    group_call_iter = group_call.loc[keep_sample]
                    if seed is not None:
                        seed += 1
                match_count[:, iter_num] = _run_one_iter(map_data_iter, mapping_median_data, other_genes, current_panel,
                                                        group_call_iter, group_distance, mapping_to_group, corr_mapping, na_str=na_str)
        
        mean_match_count = np.mean(match_count, axis=0)
        wm = np.argmax(mean_match_count)
        gene_to_add = other_genes[wm]

        if verbose:
            if optimize == "fraction_correct":
                print(f"Added {gene_to_add} with {mean_match_count[wm]:.3f}, now matching [{len(current_panel)}].")
                current_metric.append(mean_match_count[wm])
            else:
                print(f"Added {gene_to_add} with average cluster distance {-mean_match_count[wm]:.3f} [{len(current_panel)}].")
                current_metric.append(-mean_match_count[wm])
        
        current_panel.append(gene_to_add)
        
        current_panel, current_metric = build_mapping_based_marker_panel_testing(map_data=map_data, mapping_median_data=mapping_median_data, mapping_call=mapping_call, 
                                                       mapping_to_group=mapping_to_group, group_median_data=group_median_data, num_iter_each_addition=100,
                                                       panel_size=panel_size, num_subsample=num_subsample, na_str=na_str, use_parallel=use_parallel,
                                                       max_fc_gene=max_fc_gene, qmin=qmin, seed=seed, current_panel=current_panel, current_metric=current_metric,
                                                       panel_min=panel_min, verbose=verbose, corr_mapping=corr_mapping, 
                                                       optimize=optimize, group_distance=group_distance, 
                                                       cluster_genes=cluster_genes, dend=dend, percent_gene_subset=percent_gene_subset)
    return current_panel, current_metric


def _run_one_iter_parallel(map_data, group_call, mapping_call, num_subsample, seed, mapping_median_data,
                           other_genes, current_panel, group_distance,
                           mapping_to_group, corr_mapping, na_str='None'):
    keep_sample = subsample_cells(group_call, num_subsample, seed) # subsample from group_call
    map_data_iter = map_data.loc[:, keep_sample]
    group_call_iter = group_call.loc[keep_sample]
    match_count_iter = _run_one_iter(map_data_iter, mapping_median_data, other_genes, current_panel,
                                     group_call_iter, group_distance, mapping_to_group, corr_mapping,
                                     na_str=na_str)
    return match_count_iter


def _run_one_iter(map_data, mapping_median_data, other_genes, current_panel,
                  group_call, group_distance, mapping_to_group, corr_mapping, na_str='None'):
    assert na_str in mapping_to_group.index.values
    if group_distance is not None: # optimize='correlation_distance'
        assert na_str in group_distance.columns

    match_count_iter = np.zeros(len(other_genes))
    index_map = {label: idx for idx, label in enumerate(group_distance.columns)}
    group_call_inds = np.array([index_map[label] for label in group_call.values])
    
    for i, gene in enumerate(other_genes):
        ggnn = current_panel + [gene]
        if corr_mapping:
            corr_matrix_df = cor_tree_mapping(map_data=map_data, 
                                            median_data=mapping_median_data,
                                            genes_to_map=ggnn)
        else:
            corr_matrix_df = dist_tree_mapping(map_data=map_data,
                                            median_data=mapping_median_data,
                                            genes_to_map=ggnn)

        # corr_matrix_df[corr_matrix_df.isna()] = -1
        ranked_leaf_and_value = get_top_match(corr_matrix_df, replace_na_with=na_str)
        top_leaf = ranked_leaf_and_value['top_leaf'].values
        top_leaf_group = mapping_to_group.loc[top_leaf].values

        if group_distance is None: # optimize="fraction_correct"
            match_count_iter[i] = np.mean(group_call == top_leaf_group)
        else:                
            top_leaf_inds = np.array([index_map[label] for label in top_leaf_group])
            corr_dist_values = group_distance.values[group_call_inds, top_leaf_inds]
            # linear_inds = group_call_inds * len(group_labels) + top_leaf_inds # for flattening group_distance
            # corr_dist_values = group_distance[linear_inds]
            match_count_iter[i] = -np.mean(corr_dist_values)
        
    return match_count_iter



def build_mapping_based_marker_panel(map_data, median_data=None, cluster_call=None, panel_size=50, num_subsample=20, 
                                    max_fc_gene=1000, qmin=0.75, seed=None, current_panel=None, 
                                    panel_min=5, verbose=True, corr_mapping=True, 
                                    optimize="fraction_correct", cluster_distance=None, 
                                    cluster_genes=None, dend=None, percent_gene_subset=100):
    """
    Builds a panel of genes based on mapping to a reference data set.

    #CAUTION: This function is recursive, so the inputs should be a copy of the original data.

    Parameters:
    map_data (pd.DataFrame): Data to be mapped.
    median_data (pd.DataFrame): Precomputed medians. If None, it is computed.
    cluster_call (pd.Series): Cluster assignment of the columns in ref_data.
    panel_size (int): Number of genes to include in the panel.
    num_subsample (int): Number of cells to subsample from each cluster.
    max_fc_gene (int): Maximum number of genes to use for filtering.
    qmin (float): Quantile to use for filtering.
    seed (int): Random seed for reproducibility.
    current_panel (list): List of genes to start with.
    panel_min (int): Minimum number of genes to start with.
    verbose (bool): Whether to print progress messages.
    corr_mapping (bool): Whether to use correlation-based mapping.
    optimize (str): Optimization criterion for gene selection.
    cluster_distance (np.ndarray): Pairwise cluster distances.
    cluster_genes (list): List of genes to use for cluster distance calculation.
    dend (np.ndarray): Dendrogram structure for cluster distance calculation.
    percent_gene_subset (float): Percentage of genes to consider for mapping.

    Returns:
    list: List of genes in the panel.
    """

    if optimize == "dendrogram_height" and dend is None:
        return "Error: dendrogram not provided"
    
    if median_data is None:
        median_data = pd.DataFrame()
        cluster_call = pd.Series(cluster_call, index=map_data.columns)
        median_data = map_data.groupby(cluster_call, axis=1).median()
    
    if median_data.index.isnull().any():
        median_data.index = map_data.index

    if optimize == "fraction_correct":
        cluster_distance = None
    elif optimize == "correlation_distance":
        if cluster_distance is None:
            cor_dist = lambda x: 1 - np.corrcoef(x)
            if cluster_genes is None:
                cluster_genes = median_data.index
            cluster_genes = list(set(cluster_genes).intersection(set(median_data.index)))
            cluster_distance = pd.DataFrame(cor_dist(median_data.loc[cluster_genes, :].T),
                                            index=median_data.columns, columns=median_data.columns)
        
        # This commented code is for flattening the cluster_distance matrix, used at the end of the function
        # if isinstance(cluster_distance, pd.DataFrame):
        #     cluster_distance = cluster_distance.loc[median_data.columns, median_data.columns].values.flatten()
    else:
        raise ValueError("Invalid optimization criterion. Please choose 'fraction_correct' or 'correlation_distance'.")    
    # if optimize == "dendrogram_height":
    #     # Custom make_LCA_table and get_node_height functions need to be implemented
    #     lca_table = make_LCA_table(dend)
    #     cluster_distance = 1 - get_node_height(dend)[lca_table]
    #     optimize = "correlation_distance"

    # Calculate the gene expression difference between the 100th percentile cluster and the qmin percentile cluster
    # To be used for filtering (if not filtereed before)
    rank_expr_diff = rankdata(median_data.apply(lambda x: np.diff(np.percentile(x, [100 * qmin, 100])), axis=1))
    
    if median_data.shape[0] > max_fc_gene: # filter based on rank_expr_diff
        keep_genes = np.array(median_data.index)[rank_expr_diff <= max_fc_gene] # rankdata rank starts at 1
        map_data = map_data.loc[keep_genes, :]
        median_data = median_data.loc[keep_genes, :]

    panel_min = max(2, panel_min)
    if current_panel is None or len(current_panel) < panel_min:
        panel_min = max(2, panel_min - (len(current_panel) if current_panel else 0))
        current_panel = list(set(current_panel or []).union(set(median_data.index[rank_expr_diff.argsort()[:panel_min]])))
        if verbose:
            print(f"Setting starting panel as: {', '.join(current_panel)}")
    
    if len(current_panel) < panel_size:
        if num_subsample is not None:
            keep_sample = subsample_cells(cluster_call, num_subsample, seed)
            map_data = map_data.loc[:, keep_sample]
            cluster_call = cluster_call[keep_sample]
            num_subsample = None  # Once subsampled, don't subsample again in the next recursion
        
        other_genes = list(set(map_data.index).difference(set(current_panel)))
        if percent_gene_subset < 100:
            if seed is not None:
                np.random.seed(seed + len(current_panel))
            other_genes = np.random.choice(other_genes, size=int(len(other_genes) * percent_gene_subset / 100), replace=False)
        
        match_count = np.zeros(len(other_genes))
        cluster_labels = median_data.columns # for flattening cluster_distance, just in case
        # cluster_labels = cluster_distance.index.values

        # if cluster_distance is not None:
        #     assert

        index_map = {label: idx for idx, label in enumerate(cluster_labels)}
        cluster_call_inds = np.array([index_map[label] for label in cluster_call.values])
        
        for i, gene in enumerate(other_genes):
            ggnn = current_panel + [gene]
            if corr_mapping:
                corr_matrix_df = cor_tree_mapping(map_data=map_data, median_data=median_data, genes_to_map=ggnn)
            else:
                corr_matrix_df = dist_tree_mapping(map_data=map_data, median_data=median_data, genes_to_map=ggnn)

            # corr_matrix_df[corr_matrix_df.isna()] = -1
            ranked_leaf_and_value = get_top_match(corr_matrix_df)
            top_leaf = ranked_leaf_and_value['top_leaf'].values

            if cluster_distance is None: # optimize="fraction_correct"
                match_count[i] = np.mean(cluster_call == top_leaf)
            else:                
                top_leaf_inds = np.array([index_map[label] for label in top_leaf])
                corr_dist_values = cluster_distance.values[cluster_call_inds, top_leaf_inds]
                # linear_inds = cluster_call_inds * len(cluster_labels) + top_leaf_inds # for flattening cluster_distance
                # corr_dist_values = cluster_distance[linear_inds]
                match_count[i] = -np.mean(corr_dist_values)
        
        wm = np.argmax(match_count)
        gene_to_add = other_genes[wm]

        if verbose:
            if optimize == "fraction_correct":
                print(f"Added {gene_to_add} with {match_count[wm]:.3f}, now matching [{len(current_panel)}].")
            else:
                print(f"Added {gene_to_add} with average cluster distance {-match_count[wm]:.3f} [{len(current_panel)}].")
        
        current_panel.append(gene_to_add)
        current_panel = build_mapping_based_marker_panel(map_data=map_data, median_data=median_data, cluster_call=cluster_call, 
                                                       panel_size=panel_size, num_subsample=num_subsample, max_fc_gene=max_fc_gene, 
                                                       qmin=qmin, seed=seed, current_panel=current_panel, 
                                                       panel_min=panel_min, verbose=verbose, corr_mapping=corr_mapping, 
                                                       optimize=optimize, cluster_distance=cluster_distance, 
                                                       cluster_genes=cluster_genes, dend=dend, percent_gene_subset=percent_gene_subset)
    
    return current_panel


def cor_tree_mapping(map_data, median_data=None,
                     dend=None, ref_data=None, cluster_call=None, 
                     genes_to_map=None, method='pearson'):
    # Default genes_to_map to row names of map_data if not provided
    if genes_to_map is None:
        genes_to_map = map_data.index

    # If median_data is not provided
    if median_data is None:
        if cluster_call is None or ref_data is None:
            raise ValueError("Both cluster_call and ref_data must be provided if median_data is not provided.")
        # Create median_data using row-wise medians for each cluster in ref_data
        cluster_call = pd.Series(cluster_call, index=ref_data.columns)
        median_data = ref_data.groupby(cluster_call, axis=1).median()

    # If dendrogram is provided, use leaf_to_node_medians
    if dend is not None:
        # TODO: Implement leaf_to_node_medians function
        median_data = leaf_to_node_medians(dend, median_data)

    # Intersect the genes to be mapped with those in map_data and median_data
    keep_genes = list(set(genes_to_map).intersection(map_data.index).intersection(median_data.index))

    # Subset the data to include only the common genes
    map_data_subset = map_data.loc[keep_genes, :]
    median_data_subset = median_data.loc[keep_genes, :]

    # Calculate correlation matrix
    if method == 'pearson':
        corr_matrix = column_wise_corr_vectorized(map_data_subset.values, median_data_subset.values)
    elif method == 'spearman':
        corr_matrix = column_wise_spearman_corr_vectorized(map_data_subset.values, median_data_subset.values)
    else:
        raise ValueError("Invalid method. Please choose 'pearson' or 'spearman'.")
    corr_matrix_df = pd.DataFrame(corr_matrix, index=map_data.columns, columns=median_data.columns)
    return corr_matrix_df


def column_wise_corr_vectorized(A, B):
    # Subtract the mean from each column, ignoring NaN
    A_centered = A - np.nanmean(A, axis=0)
    B_centered = B - np.nanmean(B, axis=0)
    
    # Use masked arrays to ignore NaN values
    A_masked = np.ma.masked_invalid(A_centered)
    B_masked = np.ma.masked_invalid(B_centered)
    
    # Compute the dot product between A and B, ignoring NaN
    numerator = np.ma.dot(A_masked.T, B_masked)
    
    # Compute the denominator (standard deviations) for A and B
    A_var = np.ma.sum(A_masked ** 2, axis=0)
    B_var = np.ma.sum(B_masked ** 2, axis=0)
    
    denominator = np.sqrt(np.outer(A_var, B_var))
    
    # Calculate the correlation matrix (p x q)
    corr_matrix = numerator / denominator
    
    # Convert masked array back to regular array, filling any masked values with NaN
    return corr_matrix.filled(np.nan)


def column_wise_spearman_corr_vectorized(A, B):
    # Step 1: Rank the data, handling NaN values by ignoring them
    A_ranked = np.apply_along_axis(lambda x: rankdata(x, method='average', nan_policy='omit'), axis=0, arr=A)
    B_ranked = np.apply_along_axis(lambda x: rankdata(x, method='average', nan_policy='omit'), axis=0, arr=B)
    
    # Step 2: Compute the Pearson correlation on the ranked data using the previous vectorized Pearson method
    return column_wise_corr_vectorized(A_ranked, B_ranked)


# Placeholder for the leafToNodeMedians function
def leaf_to_node_medians(dend, median_data):
    # Implement this function based on your dendrogram logic
    return median_data


def dist_tree_mapping(dend=None, ref_data=None, map_data=None, median_data=None, 
                      cluster_call=None, genes_to_map=None, returnSimilarity=True, **kwargs):
    """
    Computes the Euclidean distance (or similarity) between map_data and median_data, 
    optionally leveraging a dendrogram structure for clustering.
    
    Parameters:
    dend (optional): Dendrogram structure, if available.
    ref_data (pd.DataFrame): Reference data matrix.
    map_data (pd.DataFrame): Data to be mapped. Defaults to ref_data.
    median_data (pd.DataFrame, optional): Precomputed medians. If None, it is computed.
    cluster_call (pd.Series): Cluster assignment of the columns in ref_data.
    genes_to_map (list, optional): List of genes to map.
    returnSimilarity (bool): Whether to return similarity instead of distance.
    **kwargs: Additional arguments for the distance function.
    
    Returns:
    pd.DataFrame: Distance matrix (or similarity matrix if returnSimilarity=True).
    """
    # If median_data is not provided, compute it based on cluster_call
    if median_data is None:
        if cluster_call is None or ref_data is None:
            raise ValueError("Both cluster_call and ref_data must be provided if median_data is not provided.")
        
        # Group by cluster_call and calculate row medians
        cluster_call = pd.Series(cluster_call, index=ref_data.columns)
        median_data = ref_data.groupby(cluster_call, axis=1).median()
        
        # Apply leafToNodeMedians (if dend is provided)
        if dend is not None:
            median_data = leaf_to_node_medians(dend, median_data)
    
    # Determine the intersection of genes to map
    if genes_to_map is None:
        genes_to_map = map_data.index
    keep_genes = list(set(genes_to_map).intersection(map_data.index).intersection(median_data.index))
    
    # If only one gene is selected, duplicate it to avoid single-dimensional data
    if len(keep_genes) == 1:
        keep_genes = keep_genes * 2
    
    # Subset map_data and median_data based on the intersected genes
    map_data_subset = map_data.loc[keep_genes, :].T  # Transposed for consistency with pdist usage
    median_data_subset = median_data.loc[keep_genes, :].T
    
    # Compute the Euclidean distance matrix
    eucDist = cdist(map_data_subset, median_data_subset, metric='euclidean', **kwargs)
    
    # Convert to a DataFrame for easier handling
    eucDist = pd.DataFrame(eucDist, index=map_data.columns, columns=median_data.columns)
    
    # If returnSimilarity is False, return the raw distance matrix
    if not returnSimilarity:
        return eucDist
    
    # If returnSimilarity is True, convert distance to similarity
    eucDist = np.sqrt(eucDist / np.max(eucDist.values))  # Normalize by max value
    similarity = 1 - eucDist
    
    return similarity


def get_top_match(corr_mat_df, replace_na_with='None'):
    top_leaf = corr_mat_df.idxmax(axis=1, skipna=True)
    value = corr_mat_df.max(axis=1, skipna=True)
    if replace_na_with is not None:
        top_leaf[value.isna()] = replace_na_with
        value[value.isna()] = -1
    return pd.DataFrame({'top_leaf': top_leaf, 'max_corr_value': value}, index=corr_mat_df.index)


def subsample_cells(cluster_call, num_subsample=20, seed=None):
    """
    Subsamples cells from each cluster up to a maximum number (num_subsample) for each cluster.
    Bootstrapping.
    
    Parameters:
    cluster_call (pd.Series): A Pandas Series where the index represents cell identifiers 
                          and the values represent the cluster each cell belongs to.
    num_subsample (int): The maximum number of cells to sample from each cluster.
    seed (int): The seed for random sampling (optional, for reproducibility).
    
    Returns:
    np.ndarray: Array of sampled cell indices.
    """
    
    # List to hold the sampled cell indices
    sampled_cells = []
    
    # Group cells by cluster_call and num_subsample
    for cluster_call, cell_indices in cluster_call.groupby(cluster_call):
        if seed is not None: # Set the random seed for reproducibility
            np.random.seed(seed) 
            seed += 1
        # Sample without replacement if the number of cells in the cluster is greater than num_subsample
        sampled_cells.extend(np.random.choice(cell_indices.index, num_subsample, replace=True))
    
    # Return the sampled cell indices as a NumPy array
    return np.array(sampled_cells)

# Example usage
# Assuming 'cluster_call' is a Pandas Series where index represents cell identifiers and values are cluster labels
# cluster_call = pd.Series({'cell1': 'cluster1', 'cell2': 'cluster1', 'cell3': 'cluster2', 'cell4': 'cluster2'})
# subsample_cells(cluster_call, num_subsample=1, seed=42)


# Example of how leaf_to_node_medians could be structured (to be implemented as needed)
def leaf_to_node_medians(dend, median_data):
    # Placeholder for the dendrogram-based operation on median_data
    # You need to implement this based on the logic of your dendrogram structure
    return median_data

# Example usage
# map_data = pd.DataFrame(...)  # Load or create your data
# median_data = pd.DataFrame(...)  # Load or create your data
# clusters = [...]  # Your cluster labels
# result = cor_tree_mapping(map_data, median_data, clusters=clusters)
# print(result)


############################################################################################################
### Filtering

def get_beta_score(prop_expr, return_score=True, spec_exp=2):
    """
    Calculate the beta score for each gene based on the proportion of expression values.

    Parameters:
    prop_expr (pd.DataFrame): A DataFrame of proportion of expression values.
        gene x cluster matrix.
    return_score (bool): Whether to return the beta scores or their ranks.
    spec_exp (int): Exponent for the pairwise distance calculation.

    Returns:
    np.ndarray: Array of beta scores or their ranks.
    """

    # Internal function to calculate beta score for a row
    def calc_beta(y, spec_exp=2):
        # Calculate pairwise distances
        d1 = squareform(pdist(y.reshape(-1, 1)))
        d2 = d1[np.triu_indices(d1.shape[0], 1)]
        eps1 = 1e-10  # Small value to avoid division by zero
        score1 = np.sum(d2**spec_exp) / (np.sum(d2) + eps1)
        return score1
    
    # Apply calc_beta to each row of prop_expr
    beta_score = np.apply_along_axis(calc_beta, 1, prop_expr, spec_exp)
    
    # Replace NA values (NaNs) with 0
    beta_score[np.isnan(beta_score)] = 0
    
    # Return the beta scores or their ranks
    if return_score:
        return beta_score
    else:
        score_rank = rankdata(-beta_score)  # Rank in descending order
        return score_rank


def filter_panel_genes(summary_expr, prop_expr=None, on_clusters=None, off_clusters=None, 
                       gene_lengths=None, starting_genes=None, num_binary_genes=500, 
                       min_on=10, max_on=250, max_off=50, min_length=960, 
                       max_fraction_on_clusters=0.5, on_threshold=0.5, 
                       exclude_genes=None, exclude_families=None):
    """
    Filters genes based on expression and other criteria.

    Parameters:
    summary_expr (pd.DataFrame): A DataFrame of gene expression values, usually a median.
        gene x cluster matrix.
    prop_expr (pd.DataFrame): A DataFrame of proportion of expression values.
        gene x cluster matrix.
    on_clusters (list): List of cluster names or indices to consider as 'on' clusters.
    off_clusters (list): List of cluster names or indices to consider as 'off' clusters.
    gene_lengths (np.ndarray): Array of gene lengths.
    starting_genes (list): List of genes to start with.
    num_binary_genes (int): Number of binary genes to select.
    min_on (int): Minimum expression value for max 'on' clusters.
    max_on (int): Maximum expression value for max 'on' clusters.
    max_off (int): Maximum expression value for max 'off' clusters.
    min_length (int): Minimum gene length.
    max_fraction_on_clusters (float): Maximum fraction of 'on' clusters that a gene should be expressed in.
    on_threshold (float): Threshold for 'on' expression.
    exclude_genes (list): List of genes to exclude.
    exclude_families (list): List of gene families to exclude.

    Returns:
    list: List of genes that pass the filtering criteria.
    """
    
    if starting_genes is None:
        starting_genes = ["Gad1", "Sla17a7"]
    
    if exclude_families is None:
        exclude_families = ["LOC", "LINC", "FAM", "ORF", "KIAA", "FLJ", "DKFZ", "RIK", "RPS", "RPL", "\\-"]

    # Check if summary_expr is a matrix (or DataFrame)
    if not isinstance(summary_expr, (np.ndarray, pd.DataFrame)):
        raise ValueError("summaryExpr must be a matrix or DataFrame of numeric values.")
    
    if not np.issubdtype(summary_expr.values[0, 0], np.number):
        raise ValueError("summaryExpr must contain numeric values.")
    
    if summary_expr.index is None:
        raise ValueError("Please provide summaryExpr with genes as row names.")

    if not isinstance(max_fraction_on_clusters, (int, float)):
        raise ValueError("fractionOnClusters needs to be numeric.")
    
    # If franction_on_clusters is greater than 1, assume it is in % and convert to fraction
    if max_fraction_on_clusters > 1:
        max_fraction_on_clusters /= 100
    
    genes = summary_expr.index
    genes_u = genes.str.upper()
    exclude_families = [ef.upper() for ef in exclude_families]
    
    # Create a boolean array for excluded genes and families
    exclude_genes = np.isin(genes, exclude_genes)    
    for ef in exclude_families:
        exclude_genes |= genes_u.str.contains(ef)

    # Handle on_clusters and off_clusters
    if isinstance(on_clusters, list) and all(isinstance(x, str) for x in on_clusters):
        on_clusters = np.isin(summary_expr.columns, on_clusters)
    elif isinstance(on_clusters, list) and all(isinstance(x, int) for x in on_clusters):
        on_clusters = np.isin(range(summary_expr.shape[1]), on_clusters)

    if np.sum(on_clusters) < 2:
        raise ValueError("Please provide at least two onClusters.")
    
    if off_clusters is not None:
        if isinstance(off_clusters, list) and all(isinstance(x, str) for x in off_clusters):
            off_clusters = np.isin(summary_expr.columns, off_clusters)
        elif isinstance(off_clusters, list) and all(isinstance(x, int) for x in off_clusters):
            off_clusters = np.isin(range(summary_expr.shape[1]), off_clusters)

    # Calculate max expression for on and off clusters
    max_expr_on = summary_expr.loc[:, on_clusters].max(axis=1)
    
    if off_clusters is not None:
        if np.sum(off_clusters) > 1:
            max_expr_off = summary_expr.loc[:, off_clusters].max(axis=1)
        elif np.sum(off_clusters) == 1:
            max_expr_off = summary_expr.loc[:, off_clusters]
    else:
        max_expr_off = np.full_like(max_expr_on, -np.inf)

    # Gene length validation
    if gene_lengths is not None:
        if len(gene_lengths) != len(summary_expr):
            raise ValueError("geneLengths must be of the same length as the rows of summaryExpr.")
        if not isinstance(gene_lengths, (np.ndarray, list)):
            raise ValueError("geneLengths must be numeric.")
    else:
        gene_lengths = np.full_like(max_expr_on, np.inf)

    # Filter genes
    keep_genes = (~exclude_genes) & (max_expr_on > min_on) & (max_expr_on <= max_on) & \
                 (max_expr_off <= max_off) & (gene_lengths >= min_length) & \
                 (prop_expr.loc[:, on_clusters].gt(on_threshold).mean(axis=1) <= max_fraction_on_clusters) & \
                 (prop_expr.loc[:, on_clusters].gt(on_threshold).mean(axis=1) > 0)
    filtered_out_genes = {'min_on': max_expr_on <= min_on,
                          'max_on': max_expr_on > max_on,
                          'max_off': max_expr_off > max_off,
                          'min_length': gene_lengths < min_length,
                          'max_fraction_on_clusters': prop_expr.loc[:, on_clusters].gt(on_threshold).mean(axis=1) > max_fraction_on_clusters,
                          'on_threshold': prop_expr.loc[:, on_clusters].gt(on_threshold).mean(axis=1) <= 0}
    
    keep_genes = np.nan_to_num(keep_genes, nan=False).astype(bool)

    print(f"{np.sum(keep_genes)} total genes pass constraints prior to binary score calculation.")

    # If fewer genes pass constraints than numBinaryGenes
    if np.sum(keep_genes) <= num_binary_genes:
        print(f"Warning: Fewer genes pass constraints than {num_binary_genes}, so binary score was not calculated.")
        return sorted(list(set(genes[keep_genes]).union(starting_genes)))

    # Calculate beta score (rank)
    top_beta = get_beta_score(prop_expr.loc[keep_genes, on_clusters], False)
    
    run_genes = genes[keep_genes][top_beta <= num_binary_genes]
    run_genes = sorted(list(set(run_genes).union(starting_genes)))
    
    return run_genes, filtered_out_genes



############################################################################################################
### Post-hoc analysis

def fraction_correct_with_genes(ordered_genes, map_data, median_data, cluster_call,
                                verbose=False, plot=True, return_result=True,
                                add_text=True, **kwargs):
    num_gn = range(2, len(ordered_genes))
    frac = np.zeros(len(ordered_genes))
    
    for i in num_gn:
        gns = ordered_genes[:i]
        
        # Call the Python equivalent of corTreeMapping (needs implementation)
        cor_map_tmp = cor_tree_mapping(map_data=map_data, median_data=median_data, genes_to_map=gns)
        
        # Handle NaN values by replacing them with 0
        cor_map_tmp[np.isnan(cor_map_tmp)] = 0
        
        # Call the Python equivalent of getTopMatch (needs implementation)
        top_leaf_tmp = get_top_match(cor_map_tmp)
        
        # Calculate the fraction of matches where top_leaf_tmp matches cluster_call
        frac[i] = 100 * np.mean(top_leaf_tmp.top_leaf.values == cluster_call.values)
    
    # Handle any remaining NaN values in frac
    frac[np.isnan(frac)] = 0
    
    # Plotting the result if requested
    if plot:
        ax = plot_correct_with_genes(frac, genes=ordered_genes, add_text=add_text, **kwargs)
    
    # Return the fraction array if requested
    if return_result and plot:
        return frac, ax
    elif return_result:
        return frac
    elif plot:
        return ax


def plot_correct_with_genes(frac, genes=None, ax=None, xlabel="Number of genes in panel", 
                    title_text="All clusters gene panel", ylim=(-10, 100), 
                    figsize=(10, 6), lwd=5, ylabel="Percent of cells correctly mapping", 
                    color="grey", add_text=True, **kwargs):
    # If genes are not provided, use default names (in R, names(frac) is used)
    if genes is None:
        genes = [f"Gene_{i+1}" for i in range(len(frac))]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    num_gn = np.arange(1, len(frac) + 1)

    # Plot the fraction with labels
    ax.plot(num_gn, frac, color="grey", linewidth=lwd, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title_text)
    ax.set_ylim(ylim)

    # Add horizontal dotted lines
    for h in np.arange(-2, 21) * 5:  # Equivalent to (-2:20)*5 in R
        ax.axhline(y=h, color=color, linestyle='dotted')

    # Add the horizontal solid line at h=0
    ax.axhline(y=0, color="black", linewidth=2)

    # Add text labels for the genes
    if add_text:
        for x, y, gene in zip(num_gn, frac, genes):
            ax.text(x, y, gene, rotation=90, verticalalignment='bottom')

    return ax


def plot_confusion_matrix_diff(confusion_matrix, ax=None, title_text=None, cmap="RdBu",
                          label_fontsize=20, figsize=(10, 10)):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    max_val = np.abs(confusion_matrix).max().max()
    norm = TwoSlopeNorm(vmin=-max_val, vcenter=0, vmax=max_val)


    im = ax.imshow(confusion_matrix, cmap=cmap, norm=norm)
    # add colorbar to the right, with the same height as the image
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Proportion correct', fontsize=label_fontsize)

    ax.set_xlabel(confusion_matrix.columns.name, fontsize=label_fontsize)
    ax.set_ylabel(confusion_matrix.index.name, fontsize=label_fontsize)
    ax.set_xticks(range(len(confusion_matrix.columns)))
    ax.set_yticks(range(len(confusion_matrix.index)))
    ax.set_xticklabels(confusion_matrix.columns, rotation=90)
    ax.set_yticklabels(confusion_matrix.index);
    if title_text is not None:
        ax.set_title(title_text, fontsize=label_fontsize)

    return ax


def plot_confusion_matrix(confusion_matrix, ax=None, title_text=None, cmap="viridis",
                          label_fontsize=20, figsize=(10, 10), imshow_max=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    if imshow_max is None:
        im = ax.imshow(confusion_matrix, cmap=cmap)
    else:
        im = ax.imshow(confusion_matrix, cmap=cmap, vmax=imshow_max)
    # add colorbar to the right, with the same height as the image
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Proportion correct', fontsize=label_fontsize)

    ax.set_xlabel(confusion_matrix.columns.name, fontsize=label_fontsize)
    ax.set_ylabel(confusion_matrix.index.name, fontsize=label_fontsize)
    ax.set_xticks(range(len(confusion_matrix.columns)))
    ax.set_yticks(range(len(confusion_matrix.index)))
    ax.set_xticklabels(confusion_matrix.columns, rotation=90)
    ax.set_yticklabels(confusion_matrix.index);
    if title_text is not None:
        ax.set_title(title_text, fontsize=label_fontsize)

    return ax


def get_confusion_matrix(real_cluster, predicted_cluster, proportions=True):
    # Get unique levels
    levels = np.sort(np.unique(np.concatenate((real_cluster, predicted_cluster))))

    # Convert to categorical with the same levels for both
    real_cluster = pd.Categorical(real_cluster, categories=levels, ordered=True)
    predicted_cluster = pd.Categorical(predicted_cluster, categories=levels, ordered=True)

    # Create confusion matrix using pandas crosstab
    confusion = pd.crosstab(predicted_cluster, real_cluster, rownames=['Predicted'], colnames=['Real'])

    # If proportions is True, normalize by the column sums
    if proportions:
        col_sums = confusion.sum(axis=0)
        confusion = confusion.divide(col_sums.replace(0, 1e-08), axis=1)  # pmax equivalent to avoid division by zero

    return confusion