def initialize_demo(config, env, init_sim=True):
    from omniisaacgymenvs.demos.ur10_reacher import UR10ReacherDemo

    # Mappings from strings to environments
    task_map = {
        "UR10Reacher": UR10ReacherDemo,
    }

    from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
    sim_config = SimConfig(config)

    cfg = sim_config.config
    task = task_map[cfg["task_name"]](
        name=cfg["task_name"], sim_config=sim_config, env=env
    )

    env.set_task(task=task, sim_params=sim_config.get_physics_params(), backend="torch", init_sim=init_sim)

    return task