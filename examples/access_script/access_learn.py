import coexist

# Use ACCESS to learn the simulation parameters
access = coexist.AccessScript("simulation_script.py")
access.learn(num_solutions = 10, target_sigma = 0.1, random_seed = 12345)
