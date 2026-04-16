from backend.kinova import BaseApp
import numpy as np

class Main(BaseApp):
        
    def start(self):
        self.home = False     
        
    def loop(self):        
        is_7DOF = None
        
        if(is_7DOF is None):
            raise ValueError("If you are using the big robot set is_7DOF to true. If you are using the small robot set is_7DOF to false")
        
        if(is_7DOF):
            HOME_POSITION = np.array([3.10, 5.99, 3.35, 2.54, 0.14, 4.92, 1.41])
            next_position = np.array([2.67, 5.47, 2.85, 1.92, 0.14, 4.92, 1.41])
            
        else:
            HOME_POSITION = np.array([1.75, 5.76, 2.18, 2.44, 4.54, 0.0])
            next_position = np.array([0.79, 6.11, 1.48, 1.4, 6.11, 1.57])
            
        if(self.home):
            self.kinova_robot.set_joint_angles(next_position, gripper_percentage=100)
            self.home = False

        else:
            self.kinova_robot.set_joint_angles(HOME_POSITION, gripper_percentage=0)
            self.home = True            

if __name__ == "__main__":
    simulate = None
    
    if(simulate is None):
        raise ValueError("Pick simulate or real world robot")
    
    if simulate:
        # final_project = Main(simulate=True, urdf_path="visualizer/6dof/urdf/6dof.urdf")
        # final_project = Main(simulate=True, urdf_path="visualizer/7dof/urdf/7dof.urdf")
        pass
    else:
        final_project = Main(is_suction=False)
    
    try:
        while True:
            pass
    except KeyboardInterrupt:
        final_project.shutdown()