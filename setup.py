from setuptools import setup

setup(name='cbfssm',
      version='1.0',
      packages=['cbfssm'],
      install_requires=[
          'numpy',
          'scipy',
          'sklearn',
          'tensorflow==2.5.0',
          'matplotlib',
          'tqdm'
      ])
