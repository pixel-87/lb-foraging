from lbforaging.foraging.environment import ForagingEnv as GymForagingEnv  # noqa
from lbforaging.foraging.aecEnvironment import ForagingEnv as AECForagingEnv  # noqa

# Keep the Gym-style `ForagingEnv` as the package default so `gym.make` works.
ForagingEnv = GymForagingEnv

# If you want the PettingZoo AEC implementation, import `AECForagingEnv`.
