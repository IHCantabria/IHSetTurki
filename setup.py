from setuptools import setup, find_packages

setup(
    name='IHSetTurki',
    version='1.0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'xarray',
        'numba',
        'datetime',
        'spotpy',
        'IHSetCalibration @ git+https://github.com/defreitasL/IHSetCalibration.git',
        'fast_optimization @ git+https://github.com/defreitasL/fast_optimization.git'
    ],
    author='Changbin Lim',
    author_email='limcs@unican.es',
    description='IH-SET Turki et al. (2013)',
    url='https://github.com/IHCantabria/IHSetTurki',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)