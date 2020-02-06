import setuptools

setuptools.setup(
    name='cellseg',
    version='0.0.1',
    Author='Hao Xu',
    author_email='hao.xu@scilifelab.se',
    description='initially for hpa cell segmentation',
    url='https://github.com/CellProfiling/HPA-Cell-image-segmentation/tree/hpa-image-seg',
    license='GNU',
    install_requires=[
        'click',
    ],

    packages=setuptools.find_packages(),
    zip_safe=False)