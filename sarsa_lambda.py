import gym
import numpy as np
from numpy.random import random, choice


class SarsaLambda:
    def __init__(self, gamma=0.95, lambda_p=0.5, alpha=0.05, steps=1000000, epsilon_decay=0.9995):
        self.env = gym.make("FrozenLake8x8-v0")
        self.gamma = gamma
        self.steps = steps
        self.alpha = alpha
        self.epsilon_decay = epsilon_decay  # decay per episode
        self.lambda_p = lambda_p
        self.Q = None
        self.name = 'Sarsa Lambda'

    def epsilon_greedy(self, s, Q, epsilon):
        """
        in probability epsilon, draws random action. otherwise, chooses the action that maximizes Q_s.
        Note that if Q_s contains equal values - we draw randomly among them (unlike the default behavior of np.argmax)
        """

        Q_s = Q[s, :]
        if random() < epsilon:
            return choice(len(Q_s))
        else:
            return choice(np.flatnonzero(Q_s == Q_s.max()))

    def get_action(self, s_t):
        """
        once we learned a policy using Learn function, greedily selects the best action
        """
        return np.argmax(self.Q[s_t, :])

    def monte_carlo_policy_evaluation(self, Q, n=100):
        """
        Monte Carlo policy estimation algorithm, runs n times to evaluate the value of the initial state only.
        """
        v_start = 0
        for _ in range(n):
            s_t = self.env.reset()
            done = False
            g_t = 0
            gamma_t = 1
            while not done:
                a_t = np.argmax(Q[s_t, :])
                s_t, r_t, done, info = self.env.step(a_t)
                g_t += gamma_t * r_t
                gamma_t *= self.gamma

            v_start += g_t

        return v_start / n

    def learn(self, interval=2500):
        """sarsa lambda learning algorithm. once per interval evaluates the value of the initial state."""
        no_states = self.env.observation_space.n
        no_actions = self.env.action_space.n
        Q = np.zeros((no_states, no_actions))

        x, y = [], []
        t, r, epsilon = 0, 0, 1

        while t < self.steps:
            s_prev = self.env.reset()

            E = np.zeros((no_states, no_actions))
            a_prev = self.epsilon_greedy(s_prev, Q, epsilon)
            done = False

            while not done:
                E[s_prev, a_prev] += 1

                s_t, r, done, _ = self.env.step(a_prev)
                a_t = self.epsilon_greedy(s_t, Q, epsilon)

                delta = r + self.gamma * Q[s_t, a_t] - Q[s_prev, a_prev]
                Q = Q + self.alpha * delta * E
                E = self.lambda_p * self.gamma * E

                s_prev, a_prev = s_t, a_t

                t += 1

                if t % interval == 0:
                    v = self.monte_carlo_policy_evaluation(Q)
                    y.append(v)
                    x.append(t)

            epsilon *= self.epsilon_decay

        self.Q = Q
        return x, y
