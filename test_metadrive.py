import metadrive
from metadrive.envs.metadrive_env import MetaDriveEnv

def test_metadrive():
    print("Starting MetaDrive Simulator...")
    print("Controls: W/A/S/D to drive, Q to quit.")
    
    # Configure the environment
    config = {
        "use_render": True,          # Open the 3D window
        "manual_control": True,      # Allow WASD control
        "traffic_density": 0.2,      # Add NPC vehicles (Traffic)
        "map": "CXO",                # Map with Intersections (C), Crossroads (X), and O-shapes (O)
        "window_size": (1024, 768),  # Window resolution
        "show_interface": True,      # Show speed/steering on screen
        "show_logo": False
    }
    
    env = MetaDriveEnv(config)
    
    try:
        obs, info = env.reset()
        for i in range(100000): # Run for a long time
            # In manual control mode, env.action_space.sample() is ignored
            # because the WASD keys override the input.
            action = [0.0, 0.0] 
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                print("Collision! (Auto-reset disabled, keep driving...)")
                # env.reset() # Removed so you can keep looking around
                
    except KeyboardInterrupt:
        print("Simulation stopped by user.")
    finally:
        env.close()

if __name__ == "__main__":
    test_metadrive()
