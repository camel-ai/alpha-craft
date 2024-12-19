import os
from random import choice
from getpass import getpass
import matplotlib.pyplot as plt
import torch
from open_oasis.world_model import WorldModel

from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig
from camel.messages import BaseMessage
from camel.types import ModelType, ModelPlatformType
from camel.types.enums import RoleType
from camel.models import ModelFactory

import minedojo
import numpy as np
from PIL import Image

from mcts_action.vf_agent import ValueFunctionAgent
from mcts_action.monte_carlo_tree_search import MCTS
from mcts_action.agent_node import Node

"""
Policy
"""
# Prompt for the API key securely
os.environ["OPENAI_API_KEY"] = "PUT YOUR API TOKEN HERE"

sys_msg = BaseMessage.make_assistant_message(
    role_name="Assistant",
    content="You're a helpful assistant",
)

# Set model
model=ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4O,
    model_config_dict=ChatGPTConfig(temperature=0.6).as_dict(),
)
# Set agent
vlm_agent = ChatAgent(
    sys_msg,
    model=model
)

"""
MineDojo
"""
# Add MINEDOJO_HEADLESS=1 to your environment variables to run this script headless
os.environ["MINEDOJO_HEADLESS"] = "1" 
 
env = minedojo.make(
    task_id="harvest_1_log",
    image_size=(288, 512),
    world_seed=23,
    seed=42,
)
task_prompt, task_guidance = minedojo.tasks.ALL_PROGRAMMATIC_TASK_INSTRUCTIONS['harvest_1_log']
print('task_prompt:', task_prompt)
print('task_guidance:', task_guidance)
print(f"[INFO] Create a task with prompt: {env.task_prompt}")

"""
OASIS
"""
oasis_ckpt = "path/to/oasis_checkpoint.pt"
vae_ckpt = "path/to/vae_checkpoint.pt"

world_model = WorldModel(
    oasis_ckpt=oasis_ckpt,
    vae_ckpt=vae_ckpt,
    num_frames=32,
    n_prompt_frames=1,
    ddim_steps=10,
    fps=20,
    scaling_factor=0.07843137255,
    max_noise_level=1000,
    noise_abs_max=20,
    stabilization_level=15,
    seed=0,
    device="cuda:3"
)


"""
Value Function Agent
"""
# TODO: TO BE CONFIRMED
INPUT_DESCRIPTION = ""
task_description = "{} \n {}".format(task_prompt, task_guidance)
vf_agent = ValueFunctionAgent(task_description, INPUT_DESCRIPTION)


"""
Main Loop
"""
# TODO: TO BE CONFIRMED
NUM_ACTIONS = 25
NUM_ROLLOUTS_PER_STEP = 50

def obs_to_oasis_obs(obs):
    obs_img = np.array(obs['rgb'])
    obs_img = np.transpose(obs_img, (1, 2, 0))
    return torch.tensor(obs_img).permute(2, 0, 1).unsqueeze(0).float()

# TODO: action mapping
def action_mapping(action):
    return 

obs = env.reset()
obs_img = np.array(obs['rgb'])
obs_img = np.transpose(obs_img, (1, 2, 0))
oasis_obs = obs_to_oasis_obs(obs_img)

while True:
    # NOTE: I AM ASSUMING THAT WE ARE RESETTING THE TREE PER STEP SO WE GET THE ACTUAL OBSERVATION FROM THE REAL ENVIRONMENT TO START
    # MCTS
    mcts_start_node = Node({'observation': '', 'action': -1}, task_description, NUM_ACTIONS)
    tree = MCTS(world_model, vf_agent)
    for _ in range(NUM_ROLLOUTS_PER_STEP):
        tree.do_rollout(mcts_start_node)

    # TODO: do proper action mapping
    # Get action
    action = tree.choose(mcts_start_node).state['action'] 
    
    # Take real world actions
    obs, reward, done, info = env.step(action)
    if done:
        break
