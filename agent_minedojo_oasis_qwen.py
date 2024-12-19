import os
import json
import pprint
from getpass import getpass
import matplotlib.pyplot as plt
# Set environment variable before running the script
from camel.messages import BaseMessage
from camel.types.enums import RoleType
from camel.agents import ChatAgent
from camel.configs import QwenConfig
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

# Import Agent ActionInterface
from agent.actions import ActionInterface

action_prompts = ActionInterface.get_actions_prompt()

sys_msg = BaseMessage.make_assistant_message(
    role_name="Assistant",
    content=f"You're an agent doing tasks in MineCraft environment. Here are your actions you can call:\n```{action_prompts}```\n",
)

# Set model
model = ModelFactory.create(
    model_platform=ModelPlatformType.QWEN,
    model_type=ModelType.QWEN_VL_MAX,
    model_config_dict=QwenConfig(temperature=0.2).as_dict(),
)

# Set agent
vlm_agent = ChatAgent(
    sys_msg,
    model=model
)

import minedojo
import numpy as np
from PIL import Image

# Get all available task ids
all_ids: list[str] = minedojo.tasks.ALL_TASK_IDS
# print(all_ids)

# Add MINEDOJO_HEADLESS=1 to your environment variables to run this script headless
# os.environ["MINEDOJO_HEADLESS"] = "1" 
# task_id = "harvest_1_log"
task_id = "combat_zombie_forest_iron_armors_iron_sword_shield"

env = minedojo.make(
    task_id=task_id,
    image_size=(288, 512),
    world_seed=0,
    seed=42,
)

task_prompt, task_guidance = minedojo.tasks.ALL_PROGRAMMATIC_TASK_INSTRUCTIONS['harvest_1_log']
print('task_prompt:', task_prompt)
print('task_guidance:', task_guidance)

print(f"[INFO] Create a task with prompt: {env.task_prompt}")

obs = env.reset()

obs_img = np.array(obs['rgb'])
obs_img = np.transpose(obs_img, (1, 2, 0))

image = Image.fromarray(obs_img.astype('uint8'))
# show the image
plt.imshow(image)
plt.axis("off")

# reset the agent for this demo
vlm_agent.reset()
# vlm_agent.memory.get_context()

import io
# Convert the image to a PIL.Image

img = Image.fromarray(obs_img.astype('uint8'))
img = img.convert("RGBA")
buffer = io.BytesIO()
img.save(buffer, format="PNG")
buffer.seek(0)
img = Image.open(buffer)

print('end img processing')

# Create the user message
user_msg = BaseMessage.make_user_message(
    role_name="User",
    content=(
        "Please generate actions based on the following descriptions.\n"
        "Output the actions as a JSON dictionary with the action names as keys and the action values as specified.\n\n"
        f"Your task is {task_prompt}\n"
        f"Task Guidance: {task_guidance}"
    ),
    image_list=[img],  # Include the image if applicable
)
print('start vlm_agent.step(user_msg)')
# Get the VLM agent's output
output_message = vlm_agent.step(user_msg)
action_output_str = output_message.msg.content

pprint.pprint(output_message.msg.content)