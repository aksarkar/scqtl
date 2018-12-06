import setuptools

setuptools.setup(
    name='scqtl',
    description='Single Cell QTL mapping',
    version='0.1',
    url='https://www.github.com/aksarkar/scqtl',
    author='Abhishek Sarkar',
    author_email='aksarkar@uchicago.edu',
    license='MIT',
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'tensorflow',
    ],
    packages=setuptools.find_packages(),
)
