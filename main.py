import time
import cv2
import numpy as np

# Import our custom architecture modules
from core.perception import PerceptionSystem
from core.planner import LocalPlanner
from core.controller import VehicleController

try:
    import carla
except ImportError:
    print("WARNING: carla package not found. Run 'pip install carla==0.9.15'")

def process_image(image, image_list):
    """Callback to get camera data from CARLA."""
    i = np.array(image.raw_data)
    i2 = i.reshape((image.height, image.width, 4))
    i3 = i2[:, :, :3].copy() # Make a writable copy for OpenCV
    image_list.append(i3)

def main():
    print("Starting Self-Driving Architecture Integration...")
    
    # 1. Initialize Modules
    perception = PerceptionSystem()
    planner = LocalPlanner()
    controller = VehicleController()
    
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    
    try:
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()
        
        # 2. Spawn Vehicle
        vehicle_bp = blueprint_library.filter('model3')[0]
        spawn_point = world.get_map().get_spawn_points()[0]
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        print("Vehicle spawned!")
        
        # 3. Attach Camera
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '90')
        
        # Position camera on the hood/roof
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        print("Camera attached!")
        
        image_list = []
        camera.listen(lambda image: process_image(image, image_list))
        
        # Allow time for sensors to spin up
        time.sleep(2)
        
        # 4. The Main Driving Loop
        print("Starting Autopilot Loop...")
        while True:
            if len(image_list) > 0:
                current_image = image_list.pop(0)
                
                # MODULE 1: PERCEPTION (The Eyes)
                perceived_data = perception.process_camera_frame(current_image)
                
                # Show what the camera sees
                cv2.imshow("Self-Driving Camera View", current_image)
                cv2.waitKey(1)
                
                # MODULE 2: PLANNER (The Brain)
                target_speed, target_angle = planner.determine_target(perceived_data)
                
                # MODULE 3: CONTROLLER (The Hands/Feet)
                # For this skeleton, we use fake current values (in reality, read from vehicle sensors)
                velocity = vehicle.get_velocity()
                current_speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) # km/h
                
                throttle, steer, brake = controller.get_commands(
                    target_speed=target_speed,
                    current_speed=current_speed,
                    target_angle=target_angle,
                    current_angle=0.0, # Placeholder
                    dt=0.1
                )
                
                # MODULE 4: EXECUTION
                control = carla.VehicleControl()
                control.throttle = float(throttle)
                control.steer = float(steer)
                control.brake = float(brake)
                vehicle.apply_control(control)
                
            time.sleep(0.05) # Loop delay
            
    finally:
        print("Destroying actors...")
        if 'camera' in locals():
            camera.destroy()
        if 'vehicle' in locals():
            vehicle.destroy()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
