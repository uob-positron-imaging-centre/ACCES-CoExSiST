'''ACCESS Example User Simulation Script

Must define one simulation whose parameters will be optimised this way:

    1. Use a variable called "parameters" to define this simulation's free /
       optimisable parameters. Create it using `coexist.create_parameters`.
       You can set the initial guess here.

    2. The `parameters` creation should be fully self-contained between
       `#### ACCESS PARAMETERS START` and `#### ACCESS PARAMETERS END`
       blocks (i.e. it should not depend on code ran before that).

    3. By the end of the simulation script, define two variables:
        a. `error` - one number representing this simulation's error value.
        b. `extra` - (optional) any python data structure storing extra
                     information you want to save for a simulation run (e.g.
                     particle positions).

Importantly, use `parameters.at[<free parameter name>, "value"]` to get this
simulation's free / optimisable variable values.
'''

#### ACCESS PARAMETERS START
import coexist
parameters = coexist.create_parameters(
    variables = ["cor", "separation"],
    minimums = [-3, -7],
    maximums = [+5, +3],
    values = [1, 2],               # Optional, initial guess
)

access_id = 0                       # Optional, unique ID for each simulation
#### ACCESS PARAMETERS END


# Extract variables
x = parameters.at["cor", "value"]
y = parameters.at["separation", "value"]


# Define the error value in any way - run a simulation, analyse data, etc.
a = 1
b = 100

error = (a - x) ** 2 + b * (y - x ** 2) ** 2
extra = dict(a = a, b = b, c = access_id)       # Optional
