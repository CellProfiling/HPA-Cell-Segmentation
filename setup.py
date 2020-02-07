import setuptools

requirements = []
try:
    with open('requirements.txt', 'r') as fd:
        requirements = [l.strip() for l in fd.readlines()]
except FileNotFoundError:
    print('WARNING: missing requirements.txt.')

requirements.append('pytorch_zoo@https://github.com/haoxusci/pytorch_zoo/archive/master.zip')

setuptools.setup(
    name='hpacellseg',
    version='0.1.2',
    author='Hao Xu',
    python_requires='>3.6.0',
    author_email='hao.xu@scilifelab.se',
    description='initially for hpa cell segmentation',
    url='https://github.com/CellProfiling/HPA-Cell-Segmentation',
    license='GNU',
    packages=setuptools.find_packages(),
    dependency_links=[],
    install_requires=requirements,
    include_package_data=True,
    zip_safe=False)
