from backend.kinova import BaseApp
import numpy as np
from kinova_lite_kinematics_dh import KinovaLiteDH
from utils import EndEffector

class Main(BaseApp):

    def start(self):
        self.model = KinovaLiteDH()
        self.sent = False
        self.pb = getattr(self.kinova_robot.base_kinova, "p", None)
        

    def loop(self):
        
        target_pos = [0.3, 0.0, 0.3]
        target_rpy = [0.0, np.pi/2, 0.0] 
        

        self.pb.addUserDebugPoints(
            [list(target_pos)],
            [[0, 1, 0]], # RGB 0-1
            pointSize=10,
            lifeTime=0,
        )

        target = EndEffector() # Make end effector object
        target.x, target.y, target.z = target_pos
        target.rotx, target.roty, target.rotz = target_rpy


        q_guess = np.array(self.kinova_robot.get_joint_angles(), dtype=float)
        q_sol = self.model.calc_inverse_kinematics(target, q_guess=q_guess)
        q_sol = np.asarray(q_sol, dtype=float)

        ee_check, _ = self.model.calc_forward_kinematics(q_sol.tolist())
        pos_err = np.linalg.norm([ee_check.x - target.x, 
                                  ee_check.y - target.y, 
                                  ee_check.z - target.z])
        
        print(f"[IK Result] Solved Angles: {np.round(q_sol, 3)}")
        print(f"[IK Result] Euclidean Position Error: {pos_err:.4f} m")
        print(f"End Effector Position: x={ee_check.x:.4f}, y={ee_check.y:.4f}, z={ee_check.z:.4f}")
        self.kinova_robot.set_joint_angles(q_sol, gripper_percentage=0)

if __name__ == "__main__":

    app = Main(simulate=True, urdf_path="visualizer/6dof/urdf/6dof.urdf")

    try:
        while True:
            pass
    except KeyboardInterrupt:
        app.shutdown()