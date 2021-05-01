import coexist
from coexist.schedulers import SlurmScheduler

# Use the SLURM supercomputing cluster manager to execute each simulation in
# parallel on distributed nodes.
scheduler = SlurmScheduler(
    "2:0:0",                            # Time to allocate for *one* simulation
    commands = [                        # Commands to add into the SLURM bash
        "module load Python",           # script, e.g. "module load"s needed.
        "module load PICI-LIGGGHTS",
    ],
)

# Use ACCESS to learn the simulation parameters
access = coexist.AccessScript("simulation_script.py", scheduler)
access.learn(num_solutions = 10, target_sigma = 0.1, random_seed = 12345)
