import open3d as o3d
import numpy as np
import cv2

# --------------------------
# 🔧 Configuration
# --------------------------

# Input paths
depth_path = "demo_data/honeycomb_ml/depth/depth_rect.png"  # Replace with your actual file
model_path = "honeycomb_sampled.ply"

# Camera intrinsics (from K_depth)
fx = 363.8
fy = 363.8
cx = 268.0
cy = 237.1
width = 640   # adjust if different
height = 480  # adjust if different

# Initial pose from FoundationPose
T_init = np.array([
    [-0.99125504, -0.09461696, -0.09198344,  0.85272491],
    [ 0.10821363, -0.98176634, -0.15628408, -0.01382304],
    [-0.07551914, -0.16487125,  0.98341972,  2.39968681],
    [ 0.        ,  0.        ,  0.        ,  1.        ]
])

# --------------------------
# 📦 Load the depth image
# --------------------------

depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

# Convert depth to meters if stored in mm (16-bit PNG)
if depth_raw.dtype == np.uint16:
    depth_meters = depth_raw.astype(np.float32) / 1000.0
else:
    raise ValueError("Depth image must be 16-bit PNG in mm")

# Create Open3D RGBDImage
depth_o3d = o3d.geometry.Image(depth_meters)
color_dummy = o3d.geometry.Image(np.zeros((height, width, 3), dtype=np.uint8))
rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_dummy, depth_o3d, convert_rgb_to_intensity=False
)

# --------------------------
# 🧠 Camera intrinsics
# --------------------------

intrinsics = o3d.camera.PinholeCameraIntrinsic()
intrinsics.set_intrinsics(width, height, fx, fy, cx, cy)

# --------------------------
# 🎯 Generate scene point cloud from depth
# --------------------------

target = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)

# --------------------------
# 📦 Load and transform object point cloud
# --------------------------

source = o3d.io.read_point_cloud(model_path)
source.transform(T_init)

# --------------------------
# 🔁 Run ICP
# --------------------------

threshold = 0.02  # 2 cm
reg = o3d.pipelines.registration.registration_icp(
    source, target, threshold, np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPoint()
)

print("✅ Refined pose (ICP result):")
print(reg.transformation)
o3d.io.write_point_cloud("target_scene.ply", target)           # from depth
o3d.io.write_point_cloud("source_transformed.ply", source)     # object with T_init
