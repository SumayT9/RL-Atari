from env import create_atari_env
from dqn_agent import Agent
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env = create_atari_env("PongDeterministic-v4", render=True)
model = Agent(input_size=1, action_size=env.action_space.n)
model.policy_net.load_state_dict(torch.load("dqn.pt", map_location=torch.device('cpu')).state_dict())
model.epsilon_min = 0
model.epsilon_start = 0

observation, info = env.reset()
observation = torch.from_numpy(observation).unsqueeze(0).to(device)
rewards = []
done = False
while not done:
    action = model.get_action(observation)
    next_state, reward, terminated, truncated, info = env.step(action.item())
    reward = torch.tensor([reward], device=device)
    done = terminated or truncated
    if terminated: 
        next_state = None
    else:
        next_state = torch.from_numpy(next_state).unsqueeze(0).to(device)

    observation = next_state