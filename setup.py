from setuptools import setup, find_packages


setup(
    name='spact',
    version='0.1.0',
    description='Spatially Clustered Transformer',
    url='https://github.com/ezgiogulmus/SPACT',
    author='FEO',
    author_email='',
    license='GPLv3',
    packages=find_packages(exclude=['data_processing', 'results', 'datasets_csv', "splits"]),
    install_requires=[
        "torch==2.3.0",
        "torchvision",
        "numpy==1.23.4", 
        "pandas==1.4.3", 
        "openpyxl",
        "h5py",
        "scikit-learn", 
        "scikit-survival",
        "lifelines",
        "shap",
        "tensorboardx",
        "tensorboard",
        "wandb",
        "opencv-python",
        "mmsurv @ git+https://github.com/ezgiogulmus/MMSurv",
    ],

    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: MIT",
    ]
)
