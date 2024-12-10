import os
import re
import numpy as np
import statistics
from enum import Enum, auto

from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig
from camel.messages import BaseMessage
from camel.types import ModelType, ModelPlatformType
from camel.types.enums import RoleType
from camel.models import ModelFactory

class SelfConsistencyMethod(Enum):
    MEAN = auto()
    MAJORITY_VOTE = auto()

class ValueFunctionAgent:
    def __init__(self, task_desc, input_desc, n_samples=20, temperature=1.0, top_p=1.0,
                 use_self_consistency=False,
                 self_consistency_method = SelfConsistencyMethod.MEAN):
        """
        Args:
            task_desc (str): Description of the task the agent needs to evaluate.
            input_desc (str): Description of the input for the evaluation.
            n_samples (int): Number of reasoning paths to sample for self-consistency.
            temperature (float): Sampling temperature for the model.
            top_p (float): Nucleus sampling parameter.
            use_self_consistency (bool): Whether to use self-consistency for evaluation.
        """
        assert "OPENAI_API_KEY" in os.environ and os.environ["OPENAI_API_KEY"], "Error: OPENAI_API_KEY is not set in the environment variables."

        self.task_desc = task_desc
        self.input_desc = input_desc
        self.n_samples = n_samples
        self.temperature = temperature
        self.top_p = top_p
        self.use_self_consistency = use_self_consistency
        self.self_consistency_method = self_consistency_method

        self.model = self._initialize_model()
        self._agent = self._initialize_agent()


    def _initialize_model(self):
        """
        Initialize the model with the specified configuration.
        """
        return ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O,
            model_config_dict=ChatGPTConfig(
                temperature=self.temperature,
                top_p=self.top_p
            ).as_dict(),
        )

    def _initialize_agent(self):
        """
        Initialize the chat agent with the system prompt.
        """

        prompt = f"""
            You are an expert in evaluating the performance of a player agent in Minecraft. 
            The player agent has the goal to complete a task: {self.task_desc}. 
            In each query, you will be given the following inputs: task description, action history, and observation history. 
            Your goal is to decide whether the agent's execution is successful or not. If the current state is a failure but it looks like the agent is on the right track towards success, you should also output as such.

            *IMPORTANT*
            Format your response into two lines as shown below:

            Thoughts: <your thoughts and reasoning process>
            Status: "success" or "failure"
            On the right track to success: "yes" or "no"
            """

        sys_msg = BaseMessage.make_assistant_message(
            role_name="Assistant",
            content=prompt
        )

        sys_msg = sys_msg.to_openai_system_message()

        return ChatAgent(sys_msg, model=self.model)

    def _generate_user_eval_prompt(self, act_hist):
        """
        Generate the user prompt for evaluation.

        Args:
            act_hist (list): The action history to evaluate.
        """
        return f"""
        Action History: {act_hist}
        The images corresponding to the observation history of the last {len(act_hist)} actions. The LAST IMAGE represents the current state of the player agent in the game.
        """

    def _parse_response(self, response_content):
        """
        Parse the response content and compute the score.

        Args:
            response_content (str): The content of the response message.

        Returns:
            float: The computed score (1.0 for success, 0.5 for on track, 0.0 for failure).
        """
        try:
            pred = re.search(r'Status: "?(.+)"?', response_content).group(1)
            if 'success' in pred.lower():
                return 1.0
            else:
                on_path = re.search(r'On the right track to success: "?(.+)"?', response_content).group(1)
                return 0.5 if 'yes' in on_path.lower() else 0.0
        except Exception as e:
            print(f"Error parsing response: {e}")
            return 0.0

    def eval(self, obs_hist, act_hist):
        """
        Evaluate the state using self-consistency or single-shot evaluation.

        Args:
            obs_hist (list): Observation history (e.g., image data or text descriptions).
            act_hist (list): Action history leading to the current state.

        Returns:
            float: Final value score.
        """
        # Generate the user prompt
        user_prompt = self._generate_user_prompt(act_hist)
        user_msg = BaseMessage.make_user_message(role_name="User", content=user_prompt)

        if self.use_self_consistency:
            # Self-consistency: Sample multiple reasoning paths
            all_scores = []
            for _ in range(self.n_samples):
                response = self._agent.step(user_msg)
                score = self._parse_response(response.msgs[0].content)
                all_scores.append(score)
            # Compute the final score as the mean of all sampled scores
            if self.self_consistency_method == SelfConsistencyMethod.MEAN:
                return np.mean(all_scores)
            elif self.self_consistency_method == SelfConsistencyMethod.MAJORITY_VOTE:
                return statistics.mode(all_scores)
        else:
            # Single-shot evaluation
            response = self._agent.step(user_msg)
            return self._parse_response(response.msgs[0].content)


# Usage
# agent = ValueFunctionAgent(task_desc, input_desc,
#                            use_self_consistency=True,
#                            self_consistency_method=SelfConsistencyMethod.MAJORITY_VOTE)