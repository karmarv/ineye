FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
# use an older system (18.04) to avoid opencv incompatibility (issue#3524)

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
        python3-opencv ca-certificates python3-dev git wget sudo ninja-build python3-pip
RUN ln -sv /usr/bin/python3 /usr/bin/python

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} ml -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER ml
WORKDIR /home/ml
ENV PATH="/home/ml/.local/bin:${PATH}"

# install dependencies
RUN pip install --user numpy==1.20.3
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install --user tensorboard cmake onnx   # cmake from apt-get is too old
RUN pip install --user torch==1.10 torchvision==0.11.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html

RUN pip install --user 'git+https://github.com/facebookresearch/fvcore'
# install detectron2
RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"
# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

RUN pip install --user -e detectron2_repo

# Jupyter Notebook
#
# Allow access from outside the container, and skip trying to open a browser.
# NOTE: disable authentication token for convenience. DON'T DO THIS ON A PUBLIC SERVER.
RUN pip3 --no-cache-dir install jupyter jupyterlab && \
    mkdir /home/ml/.jupyter && \
    echo "c.NotebookApp.ip = '*'" \
         "\nc.NotebookApp.open_browser = False" \
         "\nc.NotebookApp.token = ''" \
         > /home/ml/.jupyter/jupyter_notebook_config.py

EXPOSE 8888
EXPOSE 6006

# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"
WORKDIR /home/ml/det2

#CMD ["/bin/bash"]
ENTRYPOINT ["jupyter", "lab","--ip=0.0.0.0","--allow-root"]

# run detectron2 under user "appuser":
# wget http://images.cocodataset.org/val2017/000000439715.jpg -O input.jpg
# python3 demo/demo.py  \
	#--config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
	#--input input.jpg --output outputs/ \
	#--opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
