# mfishtoolspy
- Tools to analysis mFISH data, including gene panel selection from reference dataset.
- This is a python version of https://github.com/AllenInstitute/mfishtools with slight modifications
  - Bootstrapping instead of subsampling
  - Multiple iterations (default=100) per gene addition instead of single random seed to increase reliability across repetition
  - Parallel computing using dask
  - Some functions are not included yet. (e.g., fraction_correct and dendrogram_height optimization)
- This is still in active development (as of 12/10/2024)

## How to install
```
pip install mfishtoolspy
```
Or
```
git clone https://github.com/AllenNeuralDynamics/mfishtoolspy.git
cd mfishtoolspy
pip install -e .
```
