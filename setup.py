from distutils.core import setup

from setuptools import find_packages

setup(name='AtariKeras-rl',
      version='1.0',
      install_requires=['numpy', 'keras', 'keras-rl', 'gym', 'gym[atari]', 'tensorflow-gpu'],
      packages=find_packages(),
      author='Simone Barbaro',
      )
