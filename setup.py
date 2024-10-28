from setuptools import setup, find_packages

setup(
    name='your-project-name',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'Flask==3.0.0',
        'gunicorn==23.0.0',
        'numpy==1.24.3',
        'pandas==2.1.0',
        'scikit-learn==1.3.1',
        'tensorflow==2.14.0',
        'joblib==1.3.2',
    ],
)
