import setuptools

setuptools.setup(
    name='cellseg',
    author='Casper and Hao',
    version='0.0.2',
    author_email='hao.xu@scilifelab.se',
    description='initially for hpa cell segmentation',
    url='https://github.com/CellProfiling/ometif/tree/hpa-image-seg',
    license='GNU',
    install_requires=[
        'click',
    ],

    packages=setuptools.find_packages(),
    zip_safe=False)