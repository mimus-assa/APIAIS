# APIAIS
Aplication Program Interface for Artifisial Inteligence System is a set of tools that aids in the task of monitoring cameras for security purposes. 

the main two tools that have been developed for this aplication are:

API FaVe: application Program Interface for Face Verification

API ANPR: application Program Interface for Automatic Number Plate Recognition

# install


   sudo apt-get install openjdk-8-jdk git python-dev python3-dev python-numpy python3-numpy python-six python3-six build-essential python-pip python3-pip python-virtualenv swig python-wheel python3-wheel libcurl3-dev libcupti-dev


   sudo apt-get install libcurl4-openssl-dev
q

  sudo add-apt-repository ppa:graphics-drivers/ppa
    sudo apt update
    sudo apt upgrade
    ubuntu-drivers devices

sudo ubuntu-drivers autoinstall
sudo reboot
aqui
  nvidia-smi

 cd Downloads/
     sudo sh cuda_10.1.105_418.39_linux.run --override --silent --toolkit
     tar -xzvf cudnn-10.1-linux-x64-v7.6.5.32.tgz
     sudo cp cuda/include/cudnn.h /usr/local/cuda/include
     sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
     sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
     gedit ~/.bashrc



export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
export CUDA_HOME=/usr/local/cuda
  source ~/.bashrc
cd /usr/local/cuda/lib64/
sudo rm libcudnn.so.7


echo $CUDA_HOME


sudo ldconfig
cd

sh Anaconda3-2020.02-Linux-x86_64.sh

source ~/.bashrc





sudo apt-get install build-essential cmake pkg-config
sudo apt-get install libx11-dev libatlas-base-dev
sudo apt-get install libgtk-3-dev libboost-python-dev



sudo apt-get install python-dev python-pip python3-dev python3-pip
sudo -H pip2 install -U pip numpy
sudo -H pip3 install -U pip numpy


    • 
    • conda create -n tf-gpu python=3.6
    • source activate tf-gpu
    • 
conda install -c conda-forge jupyterlab

conda install -n base -c conda-forge widgetsnbextension
conda install -n tf-gpu -c conda-forge ipywidgets
conda install -c conda-forge nodejs
jupyter labextension install @jupyter-widgets/jupyterlab-manager
en base


sudo apt-get update && \
sudo apt-get install build-essential software-properties-common -y && \
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
sudo apt-get update && \
sudo apt-get install gcc-6 g++-6 -y && \
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 60 --slave /usr/bin/g++ g++ /usr/bin/g++-6 && \
gcc -v



wget http://dlib.net/files/dlib-19.17.tar.bz2
tar xvf dlib-19.17.tar.bz2
cd dlib-19.17/
mkdir build
cd build
cmake ..
cmake --build . --config Release
sudo make install
sudo ldconfig
cd 

pkg-config --libs --cflags dlib-1

source activate tf-gpu

cd dlib-19.17
python setup.py install


rm -rf dist
rm -rf tools/python/build
rm python_examples/dlib.so



pip install dlib

conda install \
tensorflow-gpu==1.14 \
cudatoolkit=10.1 \
cudnn=7.6.5 \
keras-gpu==2.2.4 \
h5py \
pillow



pip3 install face_recognition

pip3 install flask
pip3 install Flask-Bootstrap
pip3 install -U Flask-SQLAlchemy






pip3 install keras_vggface

pip3 install matplotlib

pip3 install pandas
pip3 install opencv-python

pip3 install mtcnn

pip install fuzzywuzzy[speedup]

pip3 install imutils


pip3 install sklearn
pip3 install scikit-image
pip3 install simplejson
