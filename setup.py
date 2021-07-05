from setuptools import setup

setup(
    name='eMLP',
    version='0.0.1',
    description='The electron machine learning potential package',
    author='Maarten Cools-Ceuppens',
    packages=['emlp', 'emlp/ref'],
    include_package_data=True,
    package_data = {'': ['*.so'], 'emlp/ref': ['*.*']}
    )
