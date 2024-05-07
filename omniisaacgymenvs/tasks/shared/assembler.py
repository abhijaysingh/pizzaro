from abc import abstractmethod

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omni.isaac.core.prims import RigidPrimView, XFormPrim
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage, add_reference_to_stage
from omni.isaac.core.utils.torch import *
# `scale` maps [-1, 1] to [L, U]; `unscale` maps [L, U] to [-1, 1]
from omni.isaac.core.utils.torch import scale, unscale
from omni.isaac.gym.vec_env import VecEnvBase

import numpy as np
import torch


class AssemblerTask(RLTask):
    def __init__(
        self,
        name: str,
        env: VecEnvBase,
        offset=None
    ) -> None:
        """[summary]
        """
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self.max_episode_length = self._task_cfg["env"]["episodeLength"]

        self.dist_reward_scale = self._task_cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self._task_cfg["env"]["rotRewardScale"]
        self.base_reward_scale = self._task_cfg["env"]["baseRewardScale"]
        self.workspace_reward_scale = self._task_cfg["env"]["workspaceRewardScale"]
        self.action_penalty_scale = self._task_cfg["env"]["actionPenaltyScale"]

        self.success_tolerance = self._task_cfg["env"]["successTolerance"]
        self.in_workspace_bonus = self._task_cfg["env"]["inWorkspaceBonus"]
        self.reach_goal_bonus = self._task_cfg["env"]["reachGoalBonus"]
        self.rot_eps = self._task_cfg["env"]["rotEps"]
        self.vel_obs_scale = self._task_cfg["env"]["velObsScale"]

        self.reset_dof_pos_noise = self._task_cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self._task_cfg["env"]["resetDofVelRandomInterval"]

        self.arm_dof_speed_scale = self._task_cfg["env"]["dofSpeedScale"]
        self.act_moving_average = self._task_cfg["env"]["actionsMovingAverage"]

        self.print_success_stat = self._task_cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self._task_cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self._task_cfg["env"].get("averFactor", 0.1)

        self.dt = 1.0 / 60

        RLTask.__init__(self, name, env)

        self.workspace_cube_center = torch.tensor([0.65, 0.05, 0.3], device=self.device)
        self.workspace_cube_scale = torch.tensor([0.05, 0.01, 0.05], device=self.device)
        self.workspace_cube = torch.tensor(torch.cat([self.workspace_cube_center - self.workspace_cube_scale, self.workspace_cube_center + self.workspace_cube_scale], 
                                                     dim=-1)).repeat((self.num_envs, 1))

        # Indicates which environments should be reset
        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self.av_factor = torch.tensor(self.av_factor, dtype=torch.float, device=self.device)
        self.total_successes = 0
        self.total_resets = 0
        return

    def set_up_scene(self, scene: Scene) -> None:
        self._stage = get_current_stage()
        self._assets_root_path = 'omniverse://localhost/Projects/abhijay/Isaac/2022.1'
        self.get_arm()
        self.get_topping()
        self.get_base()

        super().set_up_scene(scene)

        self._arms1, self._arms2 = self.get_arm_view(scene)
        scene.add(self._arms1)
        scene.add(self._arms2)

        self._toppings = RigidPrimView(
            prim_paths_expr="/World/envs/env_.*/object/object",
            name="object_view",
            reset_xform_properties=False,
        )
        scene.add(self._toppings)
        self._bases = RigidPrimView(
            prim_paths_expr="/World/envs/env_.*/base/object",
            name="goal_view",
            reset_xform_properties=False,
        )
        scene.add(self._bases)

    @abstractmethod
    def get_num_dof(self):
        pass

    @abstractmethod
    def get_arm(self):
        pass

    @abstractmethod
    def get_arm_view(self):
        pass

    @abstractmethod
    def get_observations(self):
        pass

    def get_topping(self):
        self.topping_start_translation = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        self.topping_start_orientation = torch.tensor([0.7071068, 0.0, 0.0, 0.7071068], device=self.device)

        self.object_usd_path = f"{self._assets_root_path}/Isaac/Props/Toppings/pepporoni_instanceable.usd"
        add_reference_to_stage(self.object_usd_path, self.default_zero_env_path + "/object")
        topping = XFormPrim(
            prim_path=self.default_zero_env_path + "/object/object",
            name="object",
            translation=self.topping_start_translation,
            orientation=self.topping_start_orientation,
            scale=self.topping_scale
        )
        self._sim_config.apply_articulation_settings("object", get_prim_at_path(topping.prim_path), self._sim_config.parse_actor_config("object"))

    def get_base(self):
        self.goal_start_translation = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        self.goal_start_orientation = torch.tensor([0.7071068, 0.0, 0.0, 0.7071068], device=self.device)

        self.goal_usd_path = f"{self._assets_root_path}/Isaac/Props/PizzaBase/pizzabase_instanceable.usd" 
        add_reference_to_stage(self.goal_usd_path, self.default_zero_env_path + "/base")
        base = XFormPrim(
            prim_path=self.default_zero_env_path + "/base/object",
            name="base",
            translation=self.goal_start_translation,
            orientation=self.goal_start_orientation,
            scale=self.base_scale
        )
        self._sim_config.apply_articulation_settings("base", get_prim_at_path(base.prim_path), self._sim_config.parse_actor_config("base"))

    def post_reset(self):
        self.num_arm_dofs = self.get_num_dof()
        self.actuated_dof_indices = torch.arange(self.num_arm_dofs, dtype=torch.long, device=self.device)

        self.arm_dof_targets = torch.zeros((self.num_envs, self._arms1.num_dof), dtype=torch.float, device=self.device)

        self.prev_targets_arm1 = torch.zeros((self.num_envs, self.num_arm_dofs), dtype=torch.float, device=self.device)
        self.cur_targets_arm1 = torch.zeros((self.num_envs, self.num_arm_dofs), dtype=torch.float, device=self.device)
        self.prev_targets_arm2 = torch.zeros((self.num_envs, self.num_arm_dofs), dtype=torch.float, device=self.device)
        self.cur_targets_arm2 = torch.zeros((self.num_envs, self.num_arm_dofs), dtype=torch.float, device=self.device)

        dof_limits = self._dof_limits
        self.arm_dof_lower_limits, self.arm_dof_upper_limits = torch.t(dof_limits[0].to(self.device))

        self.arm_dof_default_pos = torch.tensor([1.157, -1.066, -0.155, -2.239, -1.841, 1.003], dtype=torch.float, device=self.device)
        self.arm_dof_default_vel = torch.zeros(self.num_arm_dofs, dtype=torch.float, device=self.device)

        self.arm1_end_effectors_init_pos, self.arm1_end_effectors_init_rot = self._arms1._end_effectors.get_world_poses()
        self.arm2_end_effectors_init_pos, self.arm2_end_effectors_init_rot = self._arms2._end_effectors.get_world_poses()

        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self):
        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[:], self.consecutive_successes[:] = compute_arm_reward(
            self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes, self.consecutive_successes,
            self.max_episode_length, self.topping_pos, self.topping_rot, self.base_pos, self.base_rot,
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale, self.base_reward_scale,
            self.success_tolerance, self.reach_goal_bonus, self.workspace_cube, self.in_workspace_bonus,
            self.max_consecutive_successes, self.av_factor,
        )

        self.extras['consecutive_successes'] = self.consecutive_successes.mean()

        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()
            print("Direct average consecutive successes = {:.1f}".format(direct_average_successes/(self.total_resets + self.num_envs)))
            if self.total_resets > 0:
                print("Post-Reset average consecutive successes = {:.1f}".format(self.total_successes/self.total_resets))

    def pre_physics_step(self, actions):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        arm1_end_effectors_pos, arm1_end_effectors_rot = self._arms1._end_effectors.get_world_poses()
        self.topping_pos = arm1_end_effectors_pos + quat_rotate(arm1_end_effectors_rot, quat_rotate_inverse(self.arm1_end_effectors_init_rot, self.get_object_displacement_tensor()))
        self.topping_pos -= self._env_pos # subtract world env pos
        self.topping_rot = arm1_end_effectors_rot
        topping_pos = self.topping_pos + self._env_pos
        topping_rot = self.topping_rot
        self._toppings.set_world_poses(topping_pos, topping_rot)

        arm2_end_effectors_pos, arm2_end_effectors_rot = self._arms2._end_effectors.get_world_poses()
        self.base_pos = arm2_end_effectors_pos + torch.tensor([0.0, -0.1, 0.15], device=self.device)
        self.base_pos -= self._env_pos # subtract world env pos
        self.base_rot = arm2_end_effectors_rot 
        base_pos = self.base_pos + self._env_pos
        base_rot = self.base_rot
        self._bases.set_world_poses(base_pos, base_rot)

        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.actions = actions.clone().to(self.device)
        self.actions[:, 5] = 0.0
        self.actions[:, 11] = 0.0

        self.cur_targets_arm1[:, self.actuated_dof_indices] = scale(self.actions[:, :self.num_arm_dofs], 
                self.arm_dof_lower_limits[self.actuated_dof_indices], self.arm_dof_upper_limits[self.actuated_dof_indices])
        self.cur_targets_arm1[:, self.actuated_dof_indices] = self.act_moving_average * self.cur_targets_arm1[:, self.actuated_dof_indices] + \
                (1.0 - self.act_moving_average) * self.prev_targets_arm1[:, self.actuated_dof_indices]
        self.cur_targets_arm1[:, self.actuated_dof_indices] = tensor_clamp(self.cur_targets_arm1[:, self.actuated_dof_indices],
                self.arm_dof_lower_limits[self.actuated_dof_indices], self.arm_dof_upper_limits[self.actuated_dof_indices])
            
        self.cur_targets_arm2[:, self.actuated_dof_indices] = scale(self.actions[:, self.num_arm_dofs:],
                self.arm_dof_lower_limits[self.actuated_dof_indices], self.arm_dof_upper_limits[self.actuated_dof_indices])
        self.cur_targets_arm2[:, self.actuated_dof_indices] = self.act_moving_average * self.cur_targets_arm2[:, self.actuated_dof_indices] + \
                (1.0 - self.act_moving_average) * self.prev_targets_arm2[:, self.actuated_dof_indices]
        self.cur_targets_arm2[:, self.actuated_dof_indices] = tensor_clamp(self.cur_targets_arm2[:, self.actuated_dof_indices], 
                self.arm_dof_lower_limits[self.actuated_dof_indices], self.arm_dof_upper_limits[self.actuated_dof_indices])


        self.prev_targets_arm1[:, self.actuated_dof_indices] = self.cur_targets_arm1[:, self.actuated_dof_indices]
        self.prev_targets_arm2[:, self.actuated_dof_indices] = self.cur_targets_arm2[:, self.actuated_dof_indices]

        self._arms1.set_joint_position_targets(
            self.cur_targets_arm1[:, self.actuated_dof_indices], indices=None, joint_indices=self.actuated_dof_indices
        )
        self._arms2.set_joint_position_targets(
            self.cur_targets_arm2[:, self.actuated_dof_indices], indices=None, joint_indices=self.actuated_dof_indices
        )

    def is_done(self):
        self.reset_buf = torch.where(self.progress_buf >= self.max_episode_length, torch.ones_like(self.reset_buf), self.reset_buf)

    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_arm_dofs * 2 + 5), device=self.device)

        # reset arm 1
        delta_max = self.arm_dof_upper_limits - self.arm_dof_default_pos
        delta_min = self.arm_dof_lower_limits - self.arm_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * (rand_floats[:, 5:5+self.num_arm_dofs] + 1.0) * 0.5

        pos= self.arm_dof_default_pos + self.reset_dof_pos_noise * rand_delta
        dof_pos = torch.zeros((self.num_envs, self._arms1.num_dof), device=self.device)
        dof_pos[env_ids, :self.num_arm_dofs] = pos

        dof_vel = torch.zeros((self.num_envs, self._arms1.num_dof), device=self.device)
        dof_vel[env_ids, :self.num_arm_dofs] = self.arm_dof_default_vel + \
            self.reset_dof_vel_noise * rand_floats[:, 5+self.num_arm_dofs:5+self.num_arm_dofs*2]

        self.prev_targets_arm1[env_ids, :self.num_arm_dofs] = pos
        self.cur_targets_arm1[env_ids, :self.num_arm_dofs] = pos
        self.arm_dof_targets[env_ids, :self.num_arm_dofs] = pos

        self._arms1.set_joint_position_targets(self.arm_dof_targets[env_ids], indices)
        self._arms1.set_joint_positions(dof_pos[env_ids], indices)
        self._arms1.set_joint_velocities(dof_vel[env_ids], indices)

        # reset arm 2
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_arm_dofs * 2 + 5), device=self.device)
        rand_delta = delta_min + (delta_max - delta_min) * (rand_floats[:, 5:5+self.num_arm_dofs] + 1.0) * 0.8

        pos = self.arm_dof_default_pos + self.reset_dof_pos_noise * rand_delta
        dof_pos = torch.zeros((self.num_envs, self._arms1.num_dof), device=self.device)
        dof_pos[env_ids, :self.num_arm_dofs] = pos

        dof_vel = torch.zeros((self.num_envs, self._arms1.num_dof), device=self.device)
        dof_vel[env_ids, :self.num_arm_dofs] = self.arm_dof_default_vel + \
            self.reset_dof_vel_noise * rand_floats[:, 5+self.num_arm_dofs:5+self.num_arm_dofs*2]
        
        self.prev_targets_arm2[env_ids, :self.num_arm_dofs] = pos
        self.cur_targets_arm2[env_ids, :self.num_arm_dofs] = pos
        self.arm_dof_targets[env_ids, :self.num_arm_dofs] = pos
        
        self._arms2.set_joint_position_targets(self.arm_dof_targets[env_ids], indices)
        self._arms2.set_joint_positions(dof_pos[env_ids], indices)
        self._arms2.set_joint_velocities(dof_vel[env_ids], indices)

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_arm_reward(
    rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
    max_episode_length: float, topping_pos, topping_rot, base_pos, base_rot,
    dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
    actions, action_penalty_scale: float, base_reward_scale: float,
    success_tolerance: float, reach_goal_bonus: float, workspace_cube: torch.Tensor, workspace_bonus: float,
    max_consecutive_successes: int, av_factor: float
):
    # Distance between topping and base
    d = torch.norm(topping_pos - base_pos, p=2, dim=-1)
    dist_rew = d * dist_reward_scale

    # Orientaion aligment of base with world to zero radians
    quat_diff = quat_mul(base_rot, quat_conjugate(torch.tensor([0.7071068, 0.0, 0.0, 0.7071068], device=base_rot.device).repeat((base_rot.shape[0], 1))))
    base_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))
    base_rew = 1.0/(torch.abs(base_dist) + rot_eps) * base_reward_scale

    # Orientation alignment of topping with base
    quat_diff = quat_mul(topping_rot, quat_conjugate(base_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0)) 
    rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    # Action regularization
    action_penalty = torch.sum(actions ** 2, dim=-1)

    # Workspace bonus: topping is above base and both are inside workspace
    workspace_rew = torch.zeros_like(rot_rew, dtype=torch.float)
    base_in_workspace = torch.all((base_pos >= workspace_cube[:, :3]) & (base_pos <= workspace_cube[:, 3:]), dim=-1)
    workspace_rew = torch.where(base_in_workspace, workspace_rew + workspace_bonus, workspace_rew)
    topping_in_workspace = torch.all((topping_pos >= workspace_cube[:, :3]) & (topping_pos <= workspace_cube[:, 3:]), dim=-1)
    workspace_rew = torch.where(topping_in_workspace, workspace_rew + workspace_bonus, workspace_rew)
    in_workspace_above = base_in_workspace & (topping_pos[:, 2] > base_pos[:, 2]) & topping_in_workspace
    workspace_rew = torch.where(in_workspace_above, workspace_rew + (2 * workspace_bonus), workspace_rew)

    # Total reward is: distance + rotation + base + action penalty + workspace
    reward = dist_rew + rot_rew + base_rew + action_penalty * action_penalty_scale + workspace_rew
    reward = torch.where(torch.abs(d) <= success_tolerance, reward + reach_goal_bonus, reward)

    # Find out which envs hit the base and update successes count
    goal_resets = torch.where(torch.abs(d) <= success_tolerance, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    resets = reset_buf
    if max_consecutive_successes > 0:
        # Reset progress buffer on base envs if max_consecutive_successes > 0
        progress_buf = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.zeros_like(progress_buf), progress_buf)
        resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())
    
    cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

    return reward, resets, goal_resets, progress_buf, successes, cons_successes
 