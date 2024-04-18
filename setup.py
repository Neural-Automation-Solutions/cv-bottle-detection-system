from setuptools import setup, find_packages

setup(
    name='cv-bottle-detection',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'bayesian-optimization==1.4.3',
        'colorama==0.4.6',
        'joblib==1.4.0',
        'numpy==1.24.4',
        'opencv-python==4.9.0.80',
        'scikit-learn==1.3.2',
        'scipy==1.10.1',
        'threadpoolctl==3.4.0',
    ],
)
        