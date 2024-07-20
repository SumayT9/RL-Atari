import torch
import wandb
from dqn_agent import Agent
from env import create_atari_env

LOG_WANDB = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE=32

def train(n_epochs):
    torch.manual_seed(1)
    if LOG_WANDB:
        run = wandb.init(project="dqn_pong", group="experiment_1")

    env = create_atari_env("PongDeterministic-v4")
    agent = Agent(input_size=1, action_size=env.action_space.n)
    best_score = 0

    for _ in range(n_epochs):
        observation, info = env.reset()
        observation = torch.from_numpy(observation).unsqueeze(0).to(device)
        rewards = []
        done = False
        while not done:
            action = agent.get_action(observation)
            next_state, reward, terminated, truncated, info = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated
            if terminated: 
                next_state = None
            else:
                next_state = torch.from_numpy(next_state).unsqueeze(0).to(device)

            agent.memory.push(observation, action, next_state, reward)
            observation = next_state
            rewards.append(reward)
            if agent.n_steps >= BATCH_SIZE:
                agent.train_policy_net(BATCH_SIZE)
                agent.update_target_net()
                
        cum_rewards = sum(rewards)
        if LOG_WANDB:
           wandb.log({"reward": cum_rewards})
        if cum_rewards > best_score:
            best_score = cum_rewards
            torch.save(agent.policy_net, "dqn.pt")
           
    if LOG_WANDB:
        run.finish()
