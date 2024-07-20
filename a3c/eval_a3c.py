import torch
from torch.optim import Adam
from a3c.a3c_model import A3CModel
import gymnasium as gym
import torch.multiprocessing as mp
from env import create_atari_env


def eval(model_path):
    env = create_atari_env("PongDeterministic-v4",render=True)
    observation, info = env.reset()

    agent = A3CModel(input_size = 1, action_size = env.action_space.n)
    agent.load_state_dict(torch.load(model_path).state_dict())
    agent.eval()
    for i in range(3):
        done = False
        observation, info = env.reset()
        hidden = None
        while not done:
            input = torch.from_numpy(observation)
            logits, value, hidden = agent(input, hidden)

            probs = torch.softmax(logits, dim=-1)
            action = torch.argmax(probs, dim=-1).detach()
            observation, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
    env.close()
if __name__ == "__main__":
    eval("shared_saved.pt")