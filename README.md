# Reinforcement learning task: Frozen Lake (8x8) from openAI Gym

## :dart: Goal ([source](https://gym.openai.com/envs/FrozenLake8x8-v0/))
The agent controls the movement of a character in a grid world. Some tiles of the grid are walkable, and others lead to the agent falling into the water. Additionally, the movement direction of the agent is uncertain and only partially depends on the chosen direction. The agent is rewarded for finding a walkable path to a goal tile.

## :bulb: Solution
We solve the Frozen Lake problem by learning the best possible policy using Sarsa lambda algorithm.

## :clipboard: Code
At each run the code does the following: 

1. Computes the optimal policy for the different combinations of the values of alpha and lambda, using sarsa lambda algorithm. (via the ["learn" function](sarsa_lambda.py) to train the agent)
2. Displays one simulation of an episode via console
3. Plots results: each result contains 4 sub-plots displayed in one output window. Each sub-plot is based on a different combination of the values of alpha and lambda.

## :email: Contact
- rinag@post.bgu.ac.il
- schnapp@post.bgu.ac.il
