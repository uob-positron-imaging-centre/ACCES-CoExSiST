'''ACCESS Example User Simulation Script

Must define one simulation whose parameters will be optimised this way:

1. Use a variable called "parameters" to define this simulation's free /
   optimisable parameters. Create it using `coexist.create_parameters`.
   You can set the initial guess here.

2. The `parameters` creation should be *fully self-contained* between two
   `# ACCESS PARAMETERS START` and `# ACCESS PARAMETERS END` directives
   (i.e. it should not depend on code ran before that).

3. By the end of the simulation script, define a variable named ``error``
   storing a single number representing this simulation's error value.
'''

# ACCESS PARAMETERS START
import coexist
parameters = coexist.create_parameters(
    variables = ["CED", "CoR", "Epsilon", "Mu"],
    minimums = [-5, -5, -5, -5],
    maximums = [+5, +5, +5, +5],
    values = [0, 0, 0, 0],          # Optional, initial guess
)

access_id = 0                       # Optional, unique ID for each simulation
# ACCESS PARAMETERS END


# Extract variables
x = parameters["value"][0]
y = parameters["value"][1]
z = parameters["value"][2]
t = parameters["value"][3]


# Define the error value in any way - run a simulation, analyse data, etc.
error = (x**2 + y - 11)**2 + (x + y**2 - 7)**2 + z * t

