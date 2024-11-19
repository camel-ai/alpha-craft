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


## Open OASIS World Model
Please pip install this package to use the OASIS world model. 
```bash
cd open-oasis
pip install -e .
``` 

You can refer to [open-oasis/test_world_model.py](open-oasis/test_world_model.py) for the usage of the OASIS world model. Basically, `WorldModel.run(image_tensor, actions_dict_list)` would suffice for this project.