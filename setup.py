from setuptools import setup, find_packages

setup(
    name='badcad',
    version='0.1.1',
    url='https://github.com/wrongbad/badcad.git',
    author='wrongbad',
    description='csg for python/jupyter environments',
    packages=find_packages(),    
    install_requires=[
        'manifold3d',
        'pythreejs',
        'nbformat',
    ],
)