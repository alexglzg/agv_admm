from rockit import *
from casadi import *
import numpy as np

# Define problem parameters
#boat_diam = 0.30

## Problem parameters
nz    = 2                   # states to be distributed
Tf    = 2                 # control horizon [s]
Nhor  = 40                  # number of control intervals
dt    = Tf/Nhor             # sample time
number_of_robots = 1        # number of robots that are neighbors (without local)

mu = 10

## Create OCP object
ocpZ = Ocp(T=Tf)

## Variables of own robot
z_x = ocpZ.variable(grid='control',include_last=True)
z_y = ocpZ.variable(grid='control',include_last=True)
Z = vertcat(z_x,z_y)

avoid_diam = ocpZ.register_parameter(MX.sym('boat_diam', 1))

## Variables of neighbors
Z_ij = ocpZ.variable(nz*number_of_robots, grid='control',include_last=True)
#s_ij = ocpZ.variable(number_of_robots, grid='control',include_last=True)

## Parameters for copies and multipliers
lambda_i = ocpZ.register_parameter(MX.sym('lambda_i', nz), grid='control', include_last=True)
trajectory_i = ocpZ.register_parameter(MX.sym('trajectory_i', nz), grid='control', include_last=True)
lambda_ij = ocpZ.register_parameter(MX.sym('lambda_ij', nz*number_of_robots), grid='control', include_last=True)
trajectory_ij = ocpZ.register_parameter(MX.sym('trajectory_ij', nz*number_of_robots), grid='control', include_last=True)

# lambda_i = ocpZ.parameter(nz, grid='control', include_last=True)
# trajectory_i = ocpZ.parameter(nz, grid='control', include_last=True)
# lambda_ij = ocpZ.parameter(nz*number_of_robots, grid='control', include_last=True)
# trajectory_ij = ocpZ.parameter(nz*number_of_robots, grid='control', include_last=True)

c_i = Z - trajectory_i
term_i = dot(lambda_i, c_i) + mu/2*sumsqr(c_i)
if ocpZ.is_signal(term_i):
    term_i = ocpZ.sum(term_i,include_last=True)
ocpZ.add_objective(term_i)

for j in range(number_of_robots):
    Z_j = vertcat(Z_ij[2*j], Z_ij[2*j+1])
    trajectory_j = vertcat(trajectory_ij[2*j], trajectory_ij[2*j+1])
    lambda_j = vertcat(lambda_ij[2*j], lambda_ij[2*j+1])
    c_j = Z_j - trajectory_j
    term_j = dot(lambda_j, c_j) + mu/2*sumsqr(c_j)
    if ocpZ.is_signal(term_j):
        term_j = ocpZ.sum(term_j,include_last=True)
    ocpZ.add_objective(term_j)
    #ocpZ.add_objective(ocpZ.sum(s_ij[j]**2))
    distance_j = sqrt( (Z[0] - Z_j[0])**2 + (Z[1] - Z_j[1])**2 )
    #ocpZ.subject_to( (distance_j + s_ij[j]) >= (avoid_diam) )
    #ocpZ.subject_to( s_ij[j] >= 0 )
    ocpZ.subject_to( (distance_j) >= (avoid_diam) )
    
options = {"ipopt": {
    "print_level": 3,
    # "linear_solver": "ma27",
    # "tol": 1e-6,
    'sb': 'yes',
    }}
options["expand"] = True
options["print_time"] = False
options['record_time'] = True
ocpZ.solver('ipopt',options)
ocpZ.method(MultipleShooting(N=Nhor,M=1,intg='rk'))


## Set dummies for parameter values (required for ocp.to_function)
## In case you create the OCP function without solving the OCP first
ocpZ.set_value(lambda_i, DM.zeros(nz, Nhor+1))
ocpZ.set_value(trajectory_i, DM.zeros(nz, Nhor+1))
ocpZ.set_value(lambda_ij, DM.zeros(nz*number_of_robots, Nhor+1))
ocpZ.set_value(trajectory_ij, DM.zeros(nz*number_of_robots, Nhor+1))
ocpZ.set_value(avoid_diam, 0.4)

#####################################
## Define values for testing
#####################################
lambda_value = np.zeros([nz, Nhor+1])
trajectory_value = np.zeros([nz, Nhor+1])
lambda_values = np.zeros([nz*number_of_robots, Nhor+1])
trajectory_values = np.zeros([nz*number_of_robots, Nhor+1])
for j in range(number_of_robots):
    trajectory_values[2*j] = j*10+10
    trajectory_values[2*j+1] = j*10+10

#####################################
## Solve OCP with from OCP object
#####################################
## Set parameters
ocpZ.set_value(lambda_i, lambda_value)
ocpZ.set_value(lambda_ij, lambda_values)
ocpZ.set_value(trajectory_i, trajectory_value)
ocpZ.set_value(trajectory_ij, trajectory_values)

## Set initial guesses
ocpZ.set_initial(z_x, trajectory_value[0,:])
ocpZ.set_initial(z_y, trajectory_value[1,:])
ocpZ.set_initial(Z_ij, trajectory_values)

## Execute the solver
ocpZ.solve()

#########################################
## Create CasADi function from OCP object
#########################################
## Sample variables to get their symbolic representation
_, zx_samp = ocpZ.sample(z_x, grid='control')
_, zy_samp = ocpZ.sample(z_y, grid='control')
_, zij_samp = ocpZ.sample(Z_ij, grid='control')

li_samp = ocpZ.sample(lambda_i, grid='control')[1]
ti_samp = ocpZ.sample(trajectory_i, grid='control')[1]
lij_samp = ocpZ.sample(lambda_ij, grid='control')[1]
tij_samp = ocpZ.sample(trajectory_ij, grid='control')[1]

avoid_diam_samp = ocpZ.value(avoid_diam)

## Define inputs and outputs of CasADi function
initial_guesses = [zx_samp, zy_samp, zij_samp]
parameters = [li_samp, ti_samp, lij_samp, tij_samp, avoid_diam_samp]

inputs = initial_guesses + parameters
input_names = ['zx_samp', 'zy_samp', 'zij_samp', 'li_samp', 'ti_samp', 'lij_samp', 'tij_samp', 'diam_samp']

outputs = [zx_samp, zy_samp, zij_samp]
output_names = ['z_x', 'z_y', 'Z_ij']

## Create CasADi function
ocp_function = ocpZ.to_function('b_ocpZ', 
                                inputs,
                                outputs, 
                                input_names,
                                output_names)


## Serialize function
ocp_function.save('b_ocpZ.casadi')

## Test
ocp_function(trajectory_value[0,:], trajectory_value[1,:], trajectory_values, lambda_value, trajectory_value, 
              lambda_values, trajectory_values, 0.4)

'''test_solver = False

if test_solver:
    import matplotlib.pyplot as plt
    print(ocp_function)
    arg = ocp_function.convert_in(ocp_function.generate_in('ocpZ.000631.in.txt'))

    print(arg)

    res = ocp_function(**arg)

    print(res['z_x'])

    fig2, ax3 = plt.subplots()
    ax3.plot(res['z_y'].T, res['z_x'].T, 'r-')
    ax3.set_xlabel('Y [m]')
    ax3.set_ylabel('X [m]')
    fig2.tight_layout()

    plt.show()
'''