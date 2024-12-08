# AlphaCraft
AlphaCraft is a project that aims to develop a Minecraft agent that can has GPT-o1-like tree search/ self-reflection capability on top of the existing CAMEL VLM agent. The project is based on the MineDojo environment and the OASIS world model.



## TODO List

Milestone1: Task -> VLM Agent -> MCTS + OASIS world model -> Best actions (v0.1)

**Please add the dependencies packages as GIT SUBMODULES for better management/ future update.**
If you are having some modifications on the dependency repository, you can maintain a fork of the original repository and add the forked repository as a submodule. 

- [ ]  Milestone1:
    - [x]  Prepare a code base for VLM agent based on camel  @Wendong Fan @Roman Georgio
    - [ ]  Integrate with MineDojo environment @Ziyi Yang @Roman Georgio
        - [ ]  Figure out the VLM interface with MineDojo: Action space + Observation space z`
        - [ ]  Defining tasks / spotting the suitable tasks @Roman Georgio
    - [x]  Integrate OASIS world model as a function tool @Junting Chen @Haoyu
    - [ ]  Add extra component: MCTS rollout @Richie @Astitwa @Ekansh @Anubhav Kumar
        - [ ]  Generate leaf node: Call world model to get step evaluation
        - [ ]  Collect multiple leaf nodes and choose the privileged node
        - [ ]  Self-reflection (Optional)
        - [ ]  

## Installation

Please clone the repository with all the submodules by running the following command:
```bash
git clone --recurse-submodules git@github.com:camel-ai/alpha-craft.git
```

### MineDojo Environment
[MineDojo](https://github.com/MineDojo/MineDojo) is a Minecraft environment that is designed for reinforcement learning research. It is based on the Malmo platform and provides a high-level API for the agent to interact with the Minecraft world.For the full documentation of the MineDojo environment, please refer to [MineDojo Documentation](https://docs.minedojo.org/). There are options for [direct installation](https://docs.minedojo.org/sections/getting_started/install.html#direct-install) and [docker installation](https://docs.minedojo.org/sections/getting_started/install.html#docker-image) for MineDojo. Please choose the one that suits you the best.

For the direct installation, you should follow the instructions below since the **original repo is not maintained anymore**.

We strongly recommend using a conda virtual environment for AplhaCraft project if you decide to use the direct installation method.
```bash
conda create -n alpha_craft python=3.9
conda activate alpha_craft

# Install MineDojo Python package
# pip install pip==23.1 # please downgrade pip to 23.1 if you had trouble install MineDojo with pip >= 24
pip install -e MineDojo
```

The [MineDojo](./MineDojo) submodule is a forked version of the original MineDojo repository. We have made some modifications to the original repository to make it work. As a temporary solution, please specify the [MixinGradle](./MineDojo/MixinGradle-dcfaf61)'s parent directory in the [build.gradle](MineDojo/minedojo/sim/Malmo/Minecraft/build.gradle) file.
Please change the local MineDojo directory according to your own path.
```gradle
    maven {
        url = '/home/junting/repo/alpha-craft/MineDojo' // local MineDojo absolute path
    }
```

Please verify the installation by running the following command:
```bash
MINEDOJO_HEADLESS=1 python MineDojo/scripts/validate_install.py
```

### Open OASIS World Model
Please pip install this package to use the OASIS world model. 
```bash
# **Please carefully select the correct version of the torch package**
# Install pytorch (oasis is tested on cu121)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# Install the open-oasis package
pip install -e open-oasis
``` 

You can refer to [open-oasis/test_world_model.py](open-oasis/test_world_model.py) for the usage of the OASIS world model. Basically, `WorldModel.run(image_tensor, actions_dict_list)` would suffice for this project.


### Camel VLM Agent

```bash
pip install camel-ai
```