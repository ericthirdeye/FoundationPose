#!/bin/bash

# sudo is required if your user is not in the docker group
sudo docker rm -f foundationpose 2>/dev/null || true

# Remove this if you're not using GUI (or xhost isn't installed)
# xhost +

sudo docker run --gpus all -it --name foundationpose \
  --ipc=host \
  -v $HOME:/workspace \
  foundationpose

