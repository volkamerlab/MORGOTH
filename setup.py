from setuptools import setup, find_packages

setup(
    name='morgoth',
    version='1.2',
    packages=find_packages(),
    install_requires=['pandas', 'numpy',
                      'scipy', 'scikit-learn', 'multiprocess', 'fireducks; sys_platform=="linux"'],
    author='Lisa-Marie Rolli',
    author_email='lisa-marie.rolli@uni-saarland.de',
    description='Interpretable and reliable multivariate random forest for simultaneous classification and regression',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/volkamerlab/MORGOTH.git',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
)
