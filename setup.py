import setuptools

try:
    with open('requirements.txt', 'r') as fd:
        requires = [l.strip() for l in fd.readlines()]
except FileNotFoundError:
    print('requirements.txt missing')

setuptools.setup(
    name='cellseg',
    version='0.1.2',
    Author='Hao Xu',
    python_requirements='>3.6.8',
    author_email='hao.xu@scilifelab.se',
    description='initially for hpa cell segmentation',
    url='https://github.com/CellProfiling/HPA-Cell-image-segmentation',
    license='GNU',
    packages=setuptools.find_packages(),
    dependency_links=[],
    install_requires=requires,
    include_package_data=True,
    zip_safe=False)
