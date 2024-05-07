from typing import Optional

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView

import torch

class UR10View(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "UR10View",
    ) -> None:

        super().__init__(
            prim_paths_expr=prim_paths_expr,
            name=name,
            reset_xform_properties=False
        )

        if name.endswith("base_view"):
            name = "base_end_effector_view"
        else:
            name = "toppings_end_effector_view"

        self._end_effectors = RigidPrimView(prim_paths_expr=prim_paths_expr + "/ee_link", name=name, reset_xform_properties=False)

    def initialize(self, physics_sim_view):
        super().initialize(physics_sim_view)
