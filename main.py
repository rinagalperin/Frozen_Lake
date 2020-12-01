from sarsa_lambda import SarsaLambda
from simulation_runner import SimulationRunner
import matplotlib.pyplot as plt

sarsa_lambda = SarsaLambda(alpha=0.04, lambda_p=0.4)
x1, y1 = sarsa_lambda.learn()

sarsa_lambda = SarsaLambda(alpha=0.04, lambda_p=0.5)
x2, y2 = sarsa_lambda.learn()

sarsa_lambda = SarsaLambda(alpha=0.05, lambda_p=0.4)
x3, y3 = sarsa_lambda.learn()

sarsa_lambda = SarsaLambda(alpha=0.05, lambda_p=0.5)
x4, y4 = sarsa_lambda.learn()

SimulationRunner().run_simulation(sarsa_lambda)

plt.subplot(2, 2, 1)
plt.plot(x1, y1)
plt.title('alpha=0.04, lambda=0.4')
plt.xlabel("Steps Count")
plt.ylabel("Policy Value")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(x2, y2)
plt.title('alpha=0.04, lambda=0.5')
plt.xlabel("Steps Count")
plt.ylabel("Policy Value")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(x3, y3)
plt.title('alpha=0.05, lambda=0.4')
plt.xlabel("Steps Count")
plt.ylabel("Policy Value")
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(x4, y4)
plt.title('alpha=0.05, lambda=0.5')
plt.xlabel("Steps Count")
plt.ylabel("Policy Value")
plt.legend()

plt.show()
