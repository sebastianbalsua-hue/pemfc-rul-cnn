from setuptools import setup, find_packages
setup(name='pem-rul-cnn',version='0.2.0',packages=find_packages(include=['src','src.*']),
      python_requires='>=3.10',install_requires=['numpy>=1.24','pandas>=2.0','pyyaml>=6.0'],
      extras_require={'train':['tensorflow>=2.15'],'plots':['matplotlib>=3.7'],'parquet':['pyarrow>=12']})
