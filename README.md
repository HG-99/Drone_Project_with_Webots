mkdir -p ~/ws_webot_Drone
cd ~/ws_webot_Drone
docker pull cyberbotics/webots:R2025a-ubuntu22.04

xhost +local:docker

docker run -it \
  --name Webot_Drone \
  --gpus all \
  --device=/dev/dxg \
  -e DISPLAY=$DISPLAY \
  -e WAYLAND_DISPLAY=$WAYLAND_DISPLAY \
  -e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
  -e PULSE_SERVER=$PULSE_SERVER \
  -e QT_X11_NO_MITSHM=1 \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /mnt/wslg:/mnt/wslg \
  -v /usr/lib/wsl:/usr/lib/wsl \
  -v ~/ws_webot_Drone:/ws_webot_Drone \
  -w /ws_webot_Drone \
  cyberbotics/webots:R2025a-ubuntu22.04
  
apt update && apt install -y mesa-utils
glxinfo -B

apt-get install python3-pip -y
pip3 install opencv-python
pip install pyyaml


cd /ws_webot_Drone
python3 project/scripts/make_aruco_marker.py
python3 generate_world.py \
  --config project/config/config.yaml

webots


########################################################################################

# 심화 버전 : 나중에 Dockerfile 직접 수정해보고 싶을 때 진행해볼 것
git clone https://github.com/cyberbotics/webots-docker.git
cd webots-docker
docker build . --file Dockerfile \
  --tag cyberbotics/webots:latest \
  --build-arg WEBOTS_PACKAGE_PREFIX=_ubuntu-22.04

########################################################################################