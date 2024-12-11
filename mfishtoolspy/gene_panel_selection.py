import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.spatial.distance import cdist

from dask import delayed, compute
from dask.distributed import Client
import dask


def build_mapping_based_marker_panel(map_data, mapping_median_data=None, mapping_call=None, 
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
    group_median_data (pd.DataFrame): Precomputed medians for grouping. If None, it is computed.
    num_iter_each_addition (int): Number of iterations to add each gene. If num_subsample is None, this is ignored.
    panel_size (int): Number of genes to include in the panel.
    num_subsample (int): Number of cells to subsample from each group.
    na_str (str): String to replace NA values with. Used for correlation-based mapping (group_distance, mapping_to_group, def get_top_match).
    use_parallel (bool): Whether to use parallel processing for each iteration within one gene addition.
        If num_subsample is None, this is set to False.
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
            use_parallel = False

            
        match_count = np.zeros((len(other_genes), num_iter_each_addition))
        # group_labels = group_median_data.columns # for flattening group_distance, just in case
        
        
        if use_parallel:
            with Client() as client:
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
                else:
                    map_data_iter = map_data
                    group_call_iter = group_call
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
        
        current_panel, current_metric = \
            build_mapping_based_marker_panel(map_data=map_data, mapping_median_data=mapping_median_data,
                mapping_call=mapping_call, mapping_to_group=mapping_to_group,
                group_median_data=group_median_data, num_iter_each_addition=num_iter_each_addition,
                panel_size=panel_size, num_subsample=num_subsample, na_str=na_str,
                use_parallel=use_parallel, max_fc_gene=max_fc_gene, qmin=qmin, seed=seed,
                current_panel=current_panel, current_metric=current_metric,
                panel_min=panel_min, verbose=verbose, corr_mapping=corr_mapping, 
                optimize=optimize, group_distance=group_distance, cluster_genes=cluster_genes,
                dend=dend, percent_gene_subset=percent_gene_subset)
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

