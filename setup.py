from setuptools import setup, find_packages

setup(
    name='mfishtoolspy',
    version='0.0.1',
    description='Tools for mFISH data analysis and gene panel selection',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Jinho Kim',
    author_email='jinho.kim@alleninstitute.org',
    url='https://github.com/AllenNeuralDynamics/mfishtoolspy.git',
    packages=find_packages(),
    install_requires=[
        'numpy', 
        'pandas',
        'scipy',
        'matplotlib',
        'dask',
    ],
    python_requires='>=3.7',
    classifiers=[
        'Programming Language :: Python :: 3',
        # 'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    # license='MIT',
)
