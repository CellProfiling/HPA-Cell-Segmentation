import setuptools

setuptools.setup(
    name='cellseg',
    author='Casper and Hao',
<<<<<<< HEAD
    version='0.0.2.5',
=======
    version='0.0.2.1',
>>>>>>> c23a3d7222123850cb8b5879527e578120858968
    author_email='hao.xu@scilifelab.se',
    description='initially for hpa cell segmentation',
    url='https://github.com/CellProfiling/HPA-Cell-image-segmentation/tree/hpa-image-seg',
    license='GNU',
    install_requires=[
        'click',
    ],

    packages=setuptools.find_packages(),
    zip_safe=False)