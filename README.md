# PizzaRo - Multi-agent Collaboration using Reinforcement Learning

This research project explores multi-agent collaboration in completing assembly tasks. Reinforcement Learning is employed to learn a policy to place toppings on a pizza base using 2 UR10s. This project adds a UR10Assembler environment based on [OmniIsaacGymEnvs](https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs) to control a [UR10](https://www.universal-robots.com/products/ur10-robot/) with the policy learned by reinforcement learning in Omniverse Isaac Gym/Sim.

We target Isaac Sim 2022.1.1 and tested the RL code on Ubuntu 20.04. 

## Preview

![](assets/result.gif)

## Installation

Prerequisites:
- Before starting, please make sure your hardware and software meet the [system requirements](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/requirements.html#system-requirements).
- [Install Omniverse Isaac Sim 2022.1.1](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html) (Must setup Cache and Nucleus)
  - You may try out newer versions of Isaac Sim along with [their corresponding patch](https://github.com/j3soon/isaac-extended#conda-issue-on-linux), but it is not guaranteed to work.
- Double check that Nucleus is correctly installed by [following these steps](https://github.com/j3soon/isaac-extended#nucleus).
- Your computer & GPU should be able to run the Cartpole example in [OmniIsaacGymEnvs](https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs)
- (Optional) [Set up a UR3/UR5/UR10](https://www.universal-robots.com/products/) in the real world

Make sure to install Isaac Sim in the default directory and clone this repository to the home directory. Otherwise, you will encounter issues if you didn't modify the commands below accordingly.

We will use Anaconda to manage our virtual environment:

1. Clone this repository:
     ```sh
     cd ~
     git clone https://github.com/abhijaysingh/pizzaro.git
     ```
2. Generate [instanceable](https://docs.omniverse.nvidia.com/isaacsim/latest/isaac_gym_tutorials/tutorial_gym_instanceable_assets.html) UR10 assets for training:

   [Launch the Script Editor](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/tutorial_gui_interactive_scripting.html#script-editor) in Isaac Sim. Copy the content in `omniisaacgymenvs/utils/usd_utils/create_instanceable_ur10.py` and execute it inside the Script Editor window. Wait until you see the text `Done!`.

   - Copy the two folders from the `assets/models` folder to `omniverse://localhost/NVIDIA/Assets/Isaac/2022.1/Isaac/`. Check if the props and thr robots are properly copied into the specified folders. These assets will be instantiated in the training environment. 
3. (Optional) [Install ROS Melodic for Ubuntu](https://wiki.ros.org/melodic/Installation/Ubuntu) and [Set up a catkin workspace for UR10](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver/blob/master/README.md).
   
   Please change all `catkin_ws` in the commands to `ur_ws`, and make sure you can control the robot with `rqt-joint-trajectory-controller`.

   ROS support is not tested on Windows.
4. [Download and Install Anaconda](https://www.anaconda.com/products/distribution#Downloads).
   ```sh
   wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
   bash Anaconda3-2022.10-Linux-x86_64.sh
   ```
5. Patch Isaac Sim 2022.1.1
     ```sh
     export ISAAC_SIM="$HOME/.local/share/ov/pkg/isaac_sim-2022.1.1"
     cp $ISAAC_SIM/setup_python_env.sh $ISAAC_SIM/setup_python_env.sh.bak
     cp ~/pizzaro/isaac_patch/setup_python_env.sh $ISAAC_SIM/setup_python_env.sh
     ```
6. [Set up conda environment for Isaac Sim](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_python.html#advanced-running-with-anaconda)
     ```sh
     # conda remove --name isaac-sim --all
     export ISAAC_SIM="$HOME/.local/share/ov/pkg/isaac_sim-2022.1.1"
     cd $ISAAC_SIM
     conda env create -f environment.yml
     conda activate isaac-sim
     cd ~/pizzaro
     pip install -e .
     # Below is optional
     pip install pyyaml rospkg
     ```
7. Activate conda & ROS environment
     ```sh
     export ISAAC_SIM="$HOME/.local/share/ov/pkg/isaac_sim-2022.1.1"
     cd $ISAAC_SIM
     conda activate isaac-sim
     source setup_conda_env.sh
     # Below are optional
     cd ~/ur_ws
     source devel/setup.bash # or setup.zsh if you're using zsh
     ```

Please note that you should execute the commands in Step 7 for every new shell.

## Training

You can launch the training in `headless` mode as follows:

```sh
cd ~/pizzaro
python omniisaacgymenvs/scripts/rlgames_train.py task=UR10Assembler headless=True
```

The number of environments is set to 512 by default. If your GPU has small memory, you can decrease the number of environments by changing the arguments `num_envs` as below:

```sh
cd ~/pizzaro
python omniisaacgymenvs/scripts/rlgames_train.py task=UR10Assembler headless=True num_envs=512
```

You can also skip training by downloading the pre-trained model checkpoint 
from here and unzip it to `~/pizzaro/`: [model_checkpoint](https://drive.google.com/drive/folders/1K7rE8uEPoW7ihr-N11NPrDBvvadixddq?usp=drive_link)

or use this (*Recommended*) :
```sh
cd ~/pizzaro
wget https://github.com/abhijaysingh/pizzaro/releases/download/v1.0.0/runs.zip
unzip runs.zip
```


## Testing

Make sure you have model checkpoints at `~/pizzaro/runs`, you can check it with the following command:

```sh
ls ~/pizzaro/runs/UR10Assembler/nn/
```

You can visualize the learned policy by the following command:

```sh
cd ~/pizzaro
python omniisaacgymenvs/scripts/rlgames_train.py task=UR10Assembler test=True num_envs=512 checkpoint=./runs/UR10Assembler/nn/UR10Assembler.pth
```

Likewise, you can decrease the number of environments by modifying the parameter `num_envs=512`.


## Help
For any queries, please raise an issue or contact me at abhijay@umd.edu.
