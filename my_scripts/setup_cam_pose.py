import math

camera_height_offset = 0.25
orbital_radius = 0.75

orbital_angles_degrees = [45, 135, 225, 315]          # 左前、左后、右后、右前
orbital_view_names     = ["front_left", "back_left", "back_right", "front_right"]
view_index_counter = 0
# 2. 循环计算 pose（完全沿用你原来的公式）
for i, angle_deg in enumerate(orbital_angles_degrees):
    angle_rad = math.radians(angle_deg)
    
    cam_x = lookat_pos[0] + orbital_radius * math.cos(angle_rad)
    cam_y = lookat_pos[1] + orbital_radius * math.sin(angle_rad) 
    cam_z = lookat_pos[2] + camera_height_offset   
    cam_pos = (cam_x, cam_y, cam_z)
    cam.set_pose(pos=cam_pos, lookat=lookat_pos)
    cam.start_recording()