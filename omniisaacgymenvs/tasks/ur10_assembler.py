from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
from omniisaacgymenvs.tasks.shared.assembler import AssemblerTask
from omniisaacgymenvs.robots.articulations.views.ur10_view import UR10View
from omniisaacgymenvs.robots.articulations.ur10 import UR10

from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.torch import *
from omni.isaac.gym.vec_env import VecEnvBase

import numpy as np
import torch
import math


class UR10AssemblerTask(AssemblerTask):
    def __init__(
        self,
        name: str,
        sim_config: SimConfig,
        env: VecEnvBase,
        offset=None
    ) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self.num_obs_dict = {
            "full":  58,
            # 6: UR10_toppings joints position (action space)
            # 6: UR10_toppings joints velocity
            # 6: UR10_base joints position (action space)
            # 6: UR10_base joints velocity
            # 3: topping position
            # 4: topping rotation
            # 4: topping relative rotation
            # 3: base position
            # 4: base rotation
            # 4: base relative rotation
            # 12: previous action
        }

        self.topping_scale = torch.tensor([1.0] * 3)
        self.base_scale = torch.tensor([1.0] * 3)

        self._num_observations = self.num_obs_dict["full"]
        self._num_actions = 12
        self._num_states = 0

        pi = math.pi
        self._dof_limits = torch.tensor([[
                [-2*pi, 2*pi],           
                [-pi + pi/8, 0 - pi/8],  
                [-pi/2 , pi/2 ], 
                [-pi, 0],                
                [-pi, pi],               
                [-2*pi, 2*pi],           
            ]], dtype=torch.float32, device=self._cfg["sim_device"])

        AssemblerTask.__init__(self, name=name, env=env)
        return

    def get_num_dof(self):
        return self._arms1.num_dof

    def get_arm(self):
        ur10_1 = UR10(prim_path=self.default_zero_env_path + "/ur10_1", name="UR10_toppings",
                      usd_path="omniverse://localhost/Projects/abhijay/Isaac/2022.1/Isaac/Robots/UR10/ur10_toppings_instanceable.usd")
        self._sim_config.apply_articulation_settings(
            "ur10",
            get_prim_at_path(ur10_1.prim_path),
            self._sim_config.parse_actor_config("ur10"),
        )

        ur10_2 = UR10(prim_path=self.default_zero_env_path + "/ur10_2", name="UR10_base",
                      usd_path="omniverse://localhost/Projects/abhijay/Isaac/2022.1/Isaac/Robots/UR10/ur10_base_instanceable.usd",
                      translation=torch.tensor([1.5, 0.0, 0.0]))
        self._sim_config.apply_articulation_settings(
            "ur10",
            get_prim_at_path(ur10_2.prim_path),
            self._sim_config.parse_actor_config("ur10"),
        )

    def get_arm_view(self, scene):
        arm1_view = UR10View(prim_paths_expr="/World/envs/.*/ur10_1", name="ur10_toppings_view")
        scene.add(arm1_view._end_effectors)

        arm2_view = UR10View(prim_paths_expr="/World/envs/.*/ur10_2", name="ur10_base_view")
        scene.add(arm2_view._end_effectors)
        
        return arm1_view, arm2_view

    def get_object_displacement_tensor(self):
        return torch.tensor([0.0, 0.05, 0.0], device=self.device).repeat((self.num_envs, 1))

    def get_observations(self):
        self.arm1_dof_pos = self._arms1.get_joint_positions()
        self.arm1_dof_vel = self._arms1.get_joint_velocities()

        self.arm2_dof_pos = self._arms2.get_joint_positions()
        self.arm2_dof_vel = self._arms2.get_joint_velocities()

        self.compute_full_observations()

        observations = {
            self._arms1.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations

    def compute_full_observations(self):
        self.obs_buf[:, 0:self.num_arm_dofs] = unscale(self.arm1_dof_pos[:, :self.num_arm_dofs],
                self.arm_dof_lower_limits, self.arm_dof_upper_limits)
        self.obs_buf[:, self.num_arm_dofs:2*self.num_arm_dofs] = self.vel_obs_scale * self.arm1_dof_vel[:, :self.num_arm_dofs]

        self.obs_buf[:, 2*self.num_arm_dofs:3*self.num_arm_dofs] = unscale(self.arm2_dof_pos[:, :self.num_arm_dofs],
                self.arm_dof_lower_limits, self.arm_dof_upper_limits)
        self.obs_buf[:, 3*self.num_arm_dofs:4*self.num_arm_dofs] = self.vel_obs_scale * self.arm2_dof_vel[:, :self.num_arm_dofs]

        toppings = 4 * self.num_arm_dofs
        self.obs_buf[:, toppings:toppings+3] = self.topping_pos
        self.obs_buf[:, toppings+3:toppings+7] = self.topping_rot
        self.obs_buf[:, toppings+7:toppings+11] = quat_mul(self.base_rot, quat_conjugate(self.topping_rot))

        base = toppings + 11
        self.obs_buf[:, base+0:base+3] = self.base_pos
        self.obs_buf[:, base+3:base+7] = self.base_rot
        self.obs_buf[:, base+7:base+11] = quat_mul(self.topping_rot, quat_conjugate(self.base_rot))

        self.obs_buf[:, base+11:base+23] = self.actions