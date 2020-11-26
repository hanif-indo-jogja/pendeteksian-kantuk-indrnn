sudo apt -y update 
sudo apt -y upgrade

# upgrade to python3.7
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update -y
sudo apt-get install -y python3.7

sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2

sudo update-alternatives  --set python3 /usr/bin/python3.7

sudo apt -y remove python3-apt
sudo apt -y install python3-apt
sudo apt -y install python3.7-dev

sudo apt-get -y install build-essential dkms
sudo apt-get -y install freeglut3 freeglut3-dev libxi-dev libxmu-dev

# install pip3
sudo apt install -y python3-pip

# python computer vision dependencies
sudo apt install -y build-essential cmake pkg-config libx11-dev libatlas-base-dev libgtk-3-dev libboost-python-dev
sudo apt install -y libsm6 libxext6 libxrender-dev

# install python dependencies
pip3 install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install numpy scipy matplotlib pandas imutils dlib opencv-python pynvrtc
pip3 install scikit-learn==0.19.2
sudo apt -y install python3-tk

# install cuda
wget https://developer.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda-repo-ubuntu1804-10-1-local-10.1.105-418.39_1.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-10-1-local-10.1.105-418.39_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-1/7fa2af80.pub
sudo apt-get update
## include nvidia driver too
sudo apt-get -y install cuda

pip3 install cupy-cuda101

# download third party models
wget "https://docs.google.com/uc?export=download&id=1nrfc-_pdIxNn2yO1_e7CxTyJQIk3A-vX" -O third_party_models/shape_predictor_68_face_landmarks.dat