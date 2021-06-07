from setuptools import setup

setup(
    name='eMLP',
    version='0.0.1',
    description='The electron machine learning potential package',
    author='Maarten Cools-Ceuppens',
    packages=['emlp'],
    include_package_data=True,
    package_data = {'': ['*.so']}
    )

