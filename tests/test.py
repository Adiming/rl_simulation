import gym
import gym_panda
env = gym.make('panda-v0')
for i_episode in range(20):
    observation = env.reset()
    env.initialpose()
    for t in range(1000):
        env.render()
        print(observation)
        # action = env.action_space.sample()
        action = [0.7,0,0.55]
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()