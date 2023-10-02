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
