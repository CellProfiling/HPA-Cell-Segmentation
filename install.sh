conda install scikit-image imageio scipy opencv pillow=6.2.1
#conda install pytorch=1.2.0 torchvision=0.4.0 cudatoolkit=10.0 -c pytorch # use this when gpu available
conda install pytorch=1.2.0 torchvision=0.4.0 -c pytorch
pip install 'git+ssh://git@github.com:haoxusci/pytorch_zoo.git@master#egg=pytorch_zoo'

pip install . --upgrade