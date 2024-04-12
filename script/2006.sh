#!/bin/bash

# this step is to extend the rootsize of node on cloudlab
sudo env RESIZEROOT=400 ./grow-rootfs.sh

# now install all the related dependencies for gem5
sudo apt update
sudo vim /etc/apt/sources.list
#Add at the end 
deb http://dk.archive.ubuntu.com/ubuntu/ xenial main
deb http://dk.archive.ubuntu.com/ubuntu/ xenial universe
sudo apt update

sudo apt install -y {gcc,g++,gfortran}-5
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 5
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 5
sudo update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-5 5

sudo apt install -y python-pip
pip install pydot
pip install pydotplus
sudo apt-get install -y graphviz


sudo apt-get install -y python-numpy
pip install pandas
sudo apt install -y build-essential git m4 scons zlib1g zlib1g-dev libprotobuf-dev protobuf-compiler libprotoc-dev libgoogle-perftools-dev python3-dev

sudo apt install -y libhdf5-dev libpng-dev libncurses5-dev libncursesw5-dev pkg-config python3 python-is-python3 python3-pydot gawk texinfo python3-pip

sudo apt install -y linux-tools-common

sudo apt install -y linux-tools-generic

sudo apt install -y linux-cloud-tools-generic


sudo apt install -y mpi-default-* libfftw3* libpng-dev libfftw3-dev libblas-dev liblapack-dev

sudo apt install docker.io -y
sudo chmod 666 /var/run/docker.sock  

# build gem5 for a arm node.
cd simulation/gem5_06
python3 `which scons` build/ARM/gem5.fast -j$(nproc)

# build simpoint
cd ../SimPoint.3.2
make clean; make CXXFLAGS='-std=c++03 -O1'


