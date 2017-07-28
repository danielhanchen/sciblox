from distutils.core import setup

setup(
    name='sciblox',
    version='0.2.11',
    author='Daniel Han-Chen',
    author_email='danielhanchen@gmail.com',
    packages=['sciblox',],
    url='http://pypi.python.org/pypi/sciblox/',
    license='LICENSE.txt',
    description='Making data science and machine learning in Python easier.',
    long_description=open('README.txt').read(),
    install_requires=[
        "scikit-learn >= 0.18.0",
        "pandas >= 0.18.1",
        "scipy >= 0.19.0",
        "matplotlib >= 2.0.0",
        "seaborn >= 0.8.0",
        "lightgbm >= 2.0.0",
        "jupyter >= 0.9.0",
        "numpy >= 1.12.1",
        "jupyterthemes >= 0.16.0",
    ],
    extras_require = {
        'theano':  ["theano >= 0.8.0"],
        'fancyimpute': ["fancyimpute >= 0.2.0"],
        'sympy': ["sympy >= 1.1.0"],
    }
)