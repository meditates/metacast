#!/bin/bash

# this step is to extend the rootsize of node on cloudlab
sudo env RESIZEROOT=400 ./grow-rootfs.sh

# now install all the related dependencies for gem5
sudo apt update
sudo apt install screen
sudo apt install -y python3-pip

sudo apt install -y build-essential git m4 scons zlib1g zlib1g-dev libprotobuf-dev protobuf-compiler libprotoc-dev libgoogle-perftools-dev python3-dev python

sudo apt install -y libhdf5-dev libpng-dev libncurses5-dev libncursesw5-dev pkg-config python3 python3-pydot gawk texinfo -y
sudo apt install -y libblas-dev liblapack-dev

sudo apt install -y linux-tools-common

sudo apt install -y linux-tools-generic

sudo apt install -y linux-cloud-tools-generic

sudo apt install -y gcc g++ gfortran

sudo apt install -y libpng-dev libfftw3-dev libblas-dev liblapack-dev 
pip3 install pandas
sudo apt install cmake            #provides cmake
sudo apt install libx11-dev       #provides X11
sudo apt install libxt-dev
sudo apt-get install libtool-bin


sudo apt install docker.io -y
sudo chmod 666 /var/run/docker.sock  

# build gem5 for a x86 node.
cd simulation/gem5_17
python3 `which scons` build/X86/gem5.fast -j$(nproc)

# build simpoint
cd ../SimPoint.3.2
make clean; make CXXFLAGS='-std=c++03 -O1'

# install anaysis pacakges
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade Pillow

pip3 install torch torchvision
sudo apt-get install python3-pandas 
pip3 install tensorboard
pip3 install tqdm
pip3 install matplotlib
pip3 install scikit-learn
pip3 install numpy
pip3 install -U scipy
# install ax
# the ax installation need to be done in python3.10 environment
pip3 install ax-platform
pip install 'git+https://github.com/facebook/Ax.git#egg=ax-platform'
pip3 install 'git+https://github.com/cornellius-gp/gpytorch.git'
pip3 install 'git+https://github.com/pytorch/botorch.git'

