import minedojo
import numpy as np
from PIL import Image


if __name__ == "__main__":
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

    # print('env.action_space:', env.action_space)
    env.reset()

    for _ in range(100):
        action = env.action_space.no_op() # empty action
        action[0] = 1 # move forward
        action[2] = 1 # jump
        obs, reward, done, info = env.step(action)
    print('obs:', obs['rgb'])
    print('Keys of obs:', obs.keys())
    
    obs_img = np.array(obs['rgb'])
    obs_img = np.transpose(obs_img, (1, 2, 0))

    print("obs['rgb'].shape:", obs['rgb'].shape)
    print("obs['rgb'].dtype:", obs['rgb'].dtype)

    image = Image.fromarray(obs_img.astype('uint8'))

    env.close()

    image.save('output_image.png')

    # 显示图像
    image.show()