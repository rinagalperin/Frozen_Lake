import gym


class SimulationRunner:
    def __init__(self):
        self.problem = gym.make('FrozenLake8x8-v0')

    def run_simulation(self, agent):
        """"given an agent, runs simulation of the Frozen Lake problem (8x8)."""
        print('---- start simulation with {} ----'.format(agent.name))
        env = self.problem.env
        t = 0
        s_t = self.problem.reset()
        print(s_t)
        done = False
        score = 0  # reward sum
        env.render()

        while not done:
            a_t = agent.get_action(s_t)
            s_t, r_t, done, _ = env.step(a_t)
            score += r_t
            t += 1
            env.render()

        return score
