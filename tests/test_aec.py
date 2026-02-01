from lbforaging.foraging.aecEnvironment import ForagingEnv
from pettingzoo.test import api_test

def test_aec_api():
    # parameters matching a standard env configuration
    env = ForagingEnv(
        players=2,
        min_player_level=1,
        max_player_level=2,
        min_food_level=1,
        max_food_level=None, # or 1, check logic
        field_size=(8, 8),
        max_num_food=2,
        sight=8,
        max_episode_steps=50,
        force_coop=False,
        normalize_reward=True,
        grid_observation=False,
        penalty=0.0
    )
    
    # api_test checks for compliance with PettingZoo AEC API
    api_test(env, num_cycles=100, verbose_progress=False)

def test_aec_manual_cycle():
    # Simple manual check to ensure we can step through
    env = ForagingEnv(
        players=2,
        min_player_level=1,
        max_player_level=2,
        min_food_level=1,
        max_food_level=None,
        field_size=(8, 8),
        max_num_food=2,
        sight=8,
        max_episode_steps=50,
        force_coop=False
    )
    env.reset()
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if termination or truncation:
            action = None
        else:
            # Random valid action
            mask = observation["action_mask"] if isinstance(observation, dict) and "action_mask" in observation else None
            # The current observation space def in aecEnvironment doesn't seem to natively output a dict with mask for 'api_test' to pick up automatically 
            # unless wrapped. api_test handles unwrapped too usually but expects space compliance.
            # In aecEnvironment.py, _get_observation_space returns a Box.
            action = env.action_space(agent).sample()
            
            # The environment logic checks for valid actions internally but let's just sample
        
        env.step(action)
    env.close()

if __name__ == "__main__":
    try:
        print("Running AEC API Test...")
        test_aec_api()
        print("AEC API Test Passed!")
    except Exception as e:
        print(f"AEC API Test Failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        print("\nRunning Manual Cycle Test...")
        test_aec_manual_cycle()
        print("Manual Cycle Test Passed!")
    except Exception as e:
        print(f"Manual Cycle Test Failed: {e}")
        import traceback
        traceback.print_exc()
