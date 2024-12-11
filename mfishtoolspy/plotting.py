import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
from mfishtoolspy.gene_panel_selection import cor_tree_mapping, get_top_match


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


def plot_confusion_matrix_diff(confusion_matrix, ax=None, title_text=None, cmap="coolwarm",
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