# Project Navigation

My solution for the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) Navigation project, following the [Rubric](https://review.udacity.com/#!/rubrics/1889/view).

[//]: # (Image References)

[image1]: ./extra/train_episodes.png

---
## Description

The objective is to train an agent to navigate (and collect bananas) in a large, square world. It was used a `Deep Q Learning` algorithm where a 2 hidden layer deep neural network model was trained to estimate the expected reward of each action at any state.

Internally two separate networks are used for learning. The `Local` network to learn the parameters at every step and the `Target` network to estimate the target value. During every step, the experience is stored in a `Replay Buffer` memory which will be used to train the agent on every step.

## Architecture

I used a vanilla deep neural network consisting of 3 fully connected layers intermixed with ReLu activations, mapping 37 input space to 4 action states outputs with 64-dim hidden layers.

#### Hyperparameters

The initial configuration from the []() project was used and worked fine for this project, so changes were applied.

```python
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
```

In general, a big `BUFFER_SIZE` means more experience can be stored (similar to more dataset in supervised learning) and thus better training, but high memory requirements.

The `BATCH_SIZE` is important; very small values will not result in optimal learning, and very large ones can delay convergence; 64 was kept.

### Training

All training was done on the [Navigation.ipynb](Navigation.ipynb) notebook, taking as reference the DQN implemented [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn).


The training will execute the following loop until reaching the number of episodes or getting the minimal score.

```python
env_info = env.reset(train_mode=True)[brain_name] # reset the environment
state = env_info.vector_observations[0]            # get the current state
score = 0                                          # initialize the score
while True:
    action = np.random.randint(action_size)        # select an action
    env_info = env.step(action)[brain_name]        # send the action to the environment
    next_state = env_info.vector_observations[0]   # get the next state
    reward = env_info.rewards[0]                   # get the reward
    done = env_info.local_done[0]                  # see if episode has finished
    score += reward                                # update the score
    state = next_state                             # roll over the state to next time step
    if done:                                       # exit loop if episode finished
        break
```

The result (local execution) is as follows:

```python
Episode 100	Average Score: 0.66
Episode 200	Average Score: 3.39
Episode 300	Average Score: 7.03
Episode 400	Average Score: 9.55
Episode 500	Average Score: 11.78
Episode 562	Average Score: 13.07
Environment solved in 462 episodes!	Average Score: 13.07
```

A plot of rewards per episode is illustrated here:

![Training result][image1]

### Evaluation

The evaluation of the agent for a couple of episodes can be checked on [this](extra/dqt_test01.mp4) video; related to:

```python
n_episodes = 3
model_path = 'checkpoint.pth'
agent.qnetwork_local.load_state_dict(torch.load(model_path))

total_score = 0
for i_episode in range(1, n_episodes+1):
    score = 0
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0]
    while True:
        action = int(agent.act(state))
        env_s = env.step(action)
        env_info = env_s[brain_name]
        next_state, reward, done = env_info.vector_observations[0], env_info.rewards[0], env_info.local_done[0]
        score += reward
        state = next_state
        if done:
            break
    print(f"\rEpisode {i_episode} \tScore: {score}")
    total_score += score
print(f"\rAverage score in {n_episodes} episodes: {total_score/n_episodes}")
```

### Discussion and Further Work

The basic agent was implemented to solve the task. It is known that `DQN` could be unstable or even to diverge when a nonlinear function approximator such as a neural network is used to represent the action-value (Q) function.

This instability has several causes: the correlations present in the sequence of observations, the fact that small updates to Q may significantly change the policy and therefore change the data distribution, and the correlations between the action-values (Q) and the target values.

There are other alternatives to check for faster convergence to the task such as `Dueling` or `Double DQN` and to remove correlations in the observation sequences.

In particular, `Experience replay` is based on the idea that we can learn better, if we do multiple passes over the same experience, so it can be used to generate uncorrelated experience data for agent training.

### Troubleshooting

* [Error installing unityagents](https://github.com/udacity/deep-reinforcement-learning/issues/49)
* [pytorch removed from the components to install](https://github.com/udacity/deep-reinforcement-learning/issues/13)

### Resources

* [Human-level control through deep reinforcement learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
* [Rainbow DQN](https://arxiv.org/pdf/1710.02298.pdf) and [sample implementation](https://colab.research.google.com/github/Curt-Park/rainbow-is-all-you-need/blob/master/08.rainbow.ipynb#scrollTo=GMJtmlLZpyTd)
* [Dueling Network](https://arxiv.org/pdf/1511.06581.pdf)
* [Double DQN](https://arxiv.org/pdf/1509.06461.pdf)
* [Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952.pdf)
