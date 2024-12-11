from mfishtoolspy.gene_panel_selection import subsample_cells, cor_tree_mapping, dist_tree_mapping, get_top_match
from mfishtoolspy.filtering import get_beta_score
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def filter_panel_genes_archived(summary_expr, prop_expr=None, on_clusters=None, off_clusters=None, 
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
    
    return run_genes, keep_genes



def build_mapping_based_marker_panel_archived(map_data, median_data=None, cluster_call=None, panel_size=50, num_subsample=20, 
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
        current_panel = build_mapping_based_marker_panel_archived(map_data=map_data, median_data=median_data, cluster_call=cluster_call, 
                                                       panel_size=panel_size, num_subsample=num_subsample, max_fc_gene=max_fc_gene, 
                                                       qmin=qmin, seed=seed, current_panel=current_panel, 
                                                       panel_min=panel_min, verbose=verbose, corr_mapping=corr_mapping, 
                                                       optimize=optimize, cluster_distance=cluster_distance, 
                                                       cluster_genes=cluster_genes, dend=dend, percent_gene_subset=percent_gene_subset)
    
    return current_panel