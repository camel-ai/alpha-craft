import os
import re
import torch
from PIL import Image
import io

from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig
from camel.messages import BaseMessage
from camel.types import ModelType, ModelPlatformType
from camel.types.enums import RoleType
from camel.models import ModelFactory

class VLMChoose:
    @staticmethod
    def choose(model_platform=ModelPlatformType.OLLAMA, model_type='llava', model_config_dict=None, url=None):
        if model_platform == ModelPlatformType.OLLAMA:
            return ModelFactory.create(
                model_platform=ModelPlatformType.OLLAMA,
                model_type=model_type,                
                url='http://localhost:11434/v1',
            )
        if model_platform == ModelPlatformType.OPENAI:
            return ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI,
                model_type=model_type,
                model_config_dict=model_config_dict,
            )
        return None
    

class ValueFunctionAgent():
    def __init__(self, task_desc, input_desc=None):

        # assert "OPENAI_API_KEY" in os.environ and os.environ["OPENAI_API_KEY"], "Error: OPENAI_API_KEY is not set in the environment variables."
        self.task_desc = task_desc
        self.input_desc = input_desc

        sys_msg = self._init_system_prompt()

        # Set model
        # model=ModelFactory.create(
        #     model_platform=ModelPlatformType.OPENAI,
        #     model_type=ModelType.GPT_4O,
        #     model_config_dict=ChatGPTConfig(temperature=0.6).as_dict(),
        # )
        # model = VLMChoose.choose(model_platform=ModelPlatformType.OPENAI, model_type=ModelType.GPT_4O, 
        #                          model_config_dict=ChatGPTConfig(temperature=0.6).as_dict()
        #                          )
        model = VLMChoose.choose(model_platform=ModelPlatformType.OLLAMA, model_type='llava', 
                                 url='http://localhost:11434/v1',
                                 )
        # Set agent
        self._agent = ChatAgent(
            sys_msg,
            model=model
        )

    def _init_system_prompt(self):

        
        # Based on https://github.com/kohjingyu/search-agents/blob/main/agent/value_function.py#L94
        prompt = f"""
         You are an expert in evaluating the performance of a player agent in Minecraft. 
         The player agent has the goal to complete a task: {self.task_desc}. 
         In each query, you will be given the following inputs: task description, action history and observation history. 
         Your goal is to decide whether the agent's execution is successful or not. If the current state is a failure but it looks like the agent is on the right track towards success, you should also output as such.
        *IMPORTANT*
        Format your response into two lines as shown below:

        Thoughts: <your thoughts and reasoning process>
        Status: "success" or "failure"
        On the right track to success: "yes" or "no"

        """

        sys_msg = BaseMessage.make_assistant_message(
            role_name="Assistant",
            content=prompt,
        )

        return sys_msg

    def _gen_user_eval_prompt(self, act_hist):
        prompt = f"""
        Action History: {act_hist}
        The images corresponding to the observation history of the last {len(act_hist)} actions. The LAST IMAGE represents the current state of the player agent in the game.
        """
        return prompt
    def _tensor_to_image(self, tensor):
        # tensor = tensor.squeeze(0)
        # tensor = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        # return Image.fromarray(tensor)
        print("tensor.shape: ", tensor.shape, type(tensor))
        if isinstance(tensor, torch.Tensor):
            
            obs_img = tensor.squeeze().permute(1, 2, 0).numpy()
        else:
            obs_img = tensor
        print("obs_img.shape: ", obs_img.shape)
        pil_image = Image.fromarray(obs_img.astype("uint8"))
        byte_io = io.BytesIO()
        pil_image.save(byte_io, format='PNG')
        byte_io.seek(0)  # Reset pointer
        # Load it back into a PIL Image (with format set to PNG)
        image_with_format = Image.open(byte_io)
        return image_with_format
    def _obs_hist_to_images(self, obs_hist):
        return [self._tensor_to_image(obs) for obs in obs_hist]

    def eval(self, obs_hist, act_hist):
        print(f"obs_hist: {obs_hist}")
        print(f"act_hist: {act_hist}")
        print(self._gen_user_eval_prompt(act_hist))
        msg = BaseMessage.make_user_message(
            role_name= "User", content = self._gen_user_eval_prompt(act_hist), image_list=self._obs_hist_to_images(obs_hist)
        )

        response = self._agent.step(msg)

        # Get value
        response_content = response.msgs[0].content
        try:
            pred = re.search(r'Status: "?(.+)"?', response_content).group(1)
            if 'success' in pred.lower():
                score = 0.5
            else:
                score = 0.0
            # else:
            #     # Check if it's on the path to success
            #     on_path = re.search(r'On the right track to success: "?(.+)"?', response_content).group(1)
            #     if 'yes' in on_path.lower():
            #         score = 0.5
            #     else:
            #         score = 0.0
        except Exception as e:
            print(f"Error parsing response: {e}")
            score = 0.0

        return score