from setuptools import setup

setup(name='cbfssm',
      version='1.0',
      packages=['cbfssm'],
      install_requires=[
          'numpy',
          'scipy',
          'sklearn',
          'tensorflow==1.8.0',
          'matplotlib',
          'tqdm'
      ])
