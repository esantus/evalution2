from setuptools import setup

setup(
    name='evalution',
    version='2.0.0-a1',
    packages=['evalution', 'evalution.composes', 'evalution.composes.utils', 'evalution.composes.matrix',
              'evalution.composes.exception', 'evalution.composes.similarity', 'evalution.composes.composition',
              'evalution.composes.semantic_space', 'evalution.composes.transformation',
              'evalution.composes.transformation.scaling', 'evalution.composes.transformation.dim_reduction',
              'evalution.composes.transformation.feature_selection'],
    url='https://github.com/esantus/evalution2',
    license='MIT',
    author='Luca Iacoponi, Enrico Santus',
    author_email='jacoponi@gmail.com, esantus@gmail.com',
    description='A library for the evaluation of the classification of semantic relations.'
)
