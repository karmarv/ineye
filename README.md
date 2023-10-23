# INEYE
Intelligent computer vision system 


Functions:
- Multi Camera Capture (10 FPS at an interval-period of 1 Second)
- Detect object on a configured part/zone of an image 
- Detect trajectory on the detected object
- Send and archive event notifications (email or ifttt)


# Docker

Additional Instructions to Kick-off Environment

- Instructions for [NVIDIA GPU `device_id` based usage in Docker](https://docs.docker.com/config/containers/resource_constraints/#access-an-nvidia-gpu)

- Build Docker Environment
```bash
docker build -f Dockerfile.GPU --network=host --build-arg USER_ID=$UID -t ineye:v0 .
#or
docker build -f Dockerfile.CPU --network=host --build-arg USER_ID=$UID -t ineye:v0.jammy.cpu .
```
- Path environments
```bash
export DATA_PATH="./data"
export CODE_PATH="./"
```
- Docker Environment
```bash
docker run -it --gpus '"device=0"' --network=host --rm -v $(realpath $CODE_PATH):/home/ml/code -v $(realpath $DATA_PATH):/data  -p 8888:8888 -p 6006:6006 --shm-size=8gb --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --name=ineye ineye:v0
#or
docker run -it --network=host --rm -v $(realpath $CODE_PATH):/home/ml/code -v $(realpath $DATA_PATH):/data  -p 8888:8888 -p 6006:6006 --shm-size=8gb --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --name=ineye ineye:v0.jammy.cpu
```
- Preview Jupyter over forwarded ports at http://127.0.0.1:8888/lab

# Data

Sample Data taken from VIRAT Dataset (https://viratdata.org/)

# Code

```bash
conda env create -n ineye --file environment.yml --force
conda activate ineye
python viz.py --video data/VIRAT_S_010204_05_000856_000890.mp4
```

--- 

## OpenCV - Build Instructions

- Build and install
```
cmake \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX=/usr/local \
  -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules \
  -D PYTHON3_EXECUTABLE=/Users/karmax/dev/miniconda3/envs/cv/bin/python3 \
  -D PYTHON3_INCLUDE_DIR=/Users/karmax/dev/miniconda3/envs/cv/include/python3.8 \
  -D PYTHON3_NUMPY_INCLUDE_DIRS=/Users/karmax/dev/miniconda3/envs/cv/lib/python3.8/site-packages/numpy/core/include \
  -D BUILD_opencv_python2=OFF \
  -D BUILD_opencv_python3=ON \
  -D INSTALL_PYTHON_EXAMPLES=ON \
  -D OPENCV_ENABLE_NONFREE=ON \
  -D BUILD_SHARED_LIBS=ON \
  -D WITH_FFMPEG=ON \
  -D BUILD_EXAMPLES=ON ../opencv
```
```
make -j8
sudo make install 
```
- Find and copy the libraries 
```
mdfind cv2.cpython
ln -s /usr/local/lib/python3.8/site-packages/cv2/python-3.8/cv2.cpython-38-darwin.so /Users/karmax/dev/miniconda3/envs/cv/lib/python3.8/site-packages/cv2.so
```