# mfishtoolspy
- Tools to analysis mFISH data, including gene panel selection from reference dataset.
- This is a python version of https://github.com/AllenInstitute/mfishtools with slight modifications
  - Bootstrapping instead of subsampling
  - Multipel iteration (default=100) per gene addition instead of single random seed to increase reliability across repetition
  - Parallel computing using dask
  - Some functions are not included yet.
- This is still in active development (as of 12/10/2024)
