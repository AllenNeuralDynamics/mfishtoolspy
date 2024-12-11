import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata


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
    
    return run_genes, keep_genes, top_beta, filtered_out_genes


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


