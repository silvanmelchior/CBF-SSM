from setuptools import setup

setup(name='cbfssm',
      version='1.0',
      packages=['cbfssm'],
      install_requires=[
          'numpy',
          'scipy',
          'sklearn',
          'tensorflow==2.3.1',
          'matplotlib',
          'tqdm'
      ])
