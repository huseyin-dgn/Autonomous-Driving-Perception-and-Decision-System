#!/usr/bin/env bash

ADAS_DIR="$HOME/Masaüstü/Autonomous-Driving-Perception-and-Decision-System/autonomous_driving"
CARLA_DIR="$HOME/CARLA_DISK"
MODEL_PATH="$ADAS_DIR/outputs/models/bdd_yolo_v14/weights/best.pt"

echo "Eski processler temizleniyor..."
pkill -f perception_node 2>/dev/null || true
pkill -f decision_node 2>/dev/null || true
pkill -f carla_ros_bridge 2>/dev/null || true
pkill -f CarlaUE4-Linux-Shipping 2>/dev/null || true
pkill -f CarlaUE4.sh 2>/dev/null || true

sleep 2

echo "1) CARLA açılıyor..."
gnome-terminal --title="1 CARLA SERVER" -- bash -lc "
cd '$CARLA_DIR'
prime-run ./CarlaUE4.sh /Game/Carla/Maps/Town01 -quality-level=Low -ResX=840 -ResY=640 -windowed -nosound
exec bash
"

echo "2) CARLA ROS bridge açılıyor..."
gnome-terminal --title="2 CARLA ROS BRIDGE" -- bash -lc "
sleep 22
source /opt/ros/humble/setup.bash
source '$HOME/carla_ros2_ws/install/setup.bash'
ros2 launch carla_ros_bridge carla_ros_bridge.launch.py town:=Town01 timeout:=60
exec bash
"

echo "3) Ego araç + kamera + insan/araç senaryosu spawn ediliyor..."
gnome-terminal --title="3 SPAWN SCENARIO" -- bash -lc "
sleep 40
cd '$ADAS_DIR'
python3 scripts/carla_spawn_full_adas_scenario.py --spawn-ego --scenario --front-distance 16 --ped-distance 9 --ped-offset 0.5
echo ''
echo 'Senaryo scripti bitti. Kontrol:'
python3 scripts/carla_spawn_full_adas_scenario.py --status
exec bash
"

echo "4) Decision node açılıyor..."
gnome-terminal --title="4 DECISION NODE" -- bash -lc "
sleep 48
cd '$ADAS_DIR'
source /opt/ros/humble/setup.bash
source '$HOME/carla_ros2_ws/install/setup.bash'
source install/setup.bash
ros2 run autonomous_driving decision_node
exec bash
"

echo "5) Perception node açılıyor..."
gnome-terminal --title="5 PERCEPTION NODE" -- bash -lc "
sleep 55
cd '$ADAS_DIR'
source /opt/ros/humble/setup.bash
source '$HOME/carla_ros2_ws/install/setup.bash'
source install/setup.bash
ros2 run autonomous_driving perception_node \
  --ros-args \
  -r /adas/camera/front/image_raw:=/carla/ego_vehicle/rgb_front/image \
  -p model_path:='$MODEL_PATH' \
  -p sim_red_light_fallback:=false \
  -p sign_classifier_min_conf:=0.60
exec bash
"

echo "6) Kontrol terminali açılıyor..."
gnome-terminal --title="6 CONTROL" -- bash -lc "
sleep 65
cd '$ADAS_DIR'
source /opt/ros/humble/setup.bash
source '$HOME/carla_ros2_ws/install/setup.bash'
source install/setup.bash

echo '--- TOPICS ---'
ros2 topic list | grep -E 'carla|adas|ego|image|decision' || true

echo ''
echo '--- CAMERA HZ ---'
timeout 8 ros2 topic hz /carla/ego_vehicle/rgb_front/image || true

echo ''
echo '--- PERCEPTION HZ ---'
timeout 8 ros2 topic hz /adas/perception/annotated_image || true

echo ''
echo '--- DETECTION JSON ---'
ros2 topic echo /adas/perception/detections_json --once --field data || true

echo ''
echo '--- DECISION JSON ---'
ros2 topic echo /adas/decision --once --field data || true

exec bash
"

echo ""
echo "Sistem başlatıldı."
echo "60-70 saniye bekle."
echo "Sonra ADAS debug penceresine bak."
