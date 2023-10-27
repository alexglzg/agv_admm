from rockit import *
from casadi import *
import numpy as np

# Define problem parameters
u_max_speed = 0.5
u_max_rotation = 0.5

# Problem parameters
nx    = 3                   # the system is composed of 3 states per robot
nz    = 2                   # states to be distributed
nu    = 2                   # the system has 2 inputs per robot
Tf    = 2                   # control horizon [s]
Nhor  = 40                  # number of control intervals
dt    = Tf/Nhor             # sample time
number_of_robots = 1        # number of robots that are neighbors (without local)

mu = 10

# Create OCP object
ocpX = Ocp(T=Tf)

# States
x = ocpX.state()
y = ocpX.state()
psi = ocpX.state()
X = vertcat(x,y)
Xx = vertcat(x,y,psi)

# Controls
v = ocpX.control()
w = ocpX.control()

# Derivatives
ocpX.set_der(x, v*cos(psi))
ocpX.set_der(y, v*sin(psi))
ocpX.set_der(psi, w)

# Reference parameters
x_ref = ocpX.register_parameter(MX.sym('x_ref', 1))
y_ref = ocpX.register_parameter(MX.sym('y_ref', 1))

# Lagrange objective
distance = sqrt( (x_ref - x)**2 + (y_ref - y)**2 )
ocpX.add_objective(ocpX.sum((distance)**2))
ocpX.add_objective(ocpX.at_tf((distance)**2))
ocpX.add_objective(ocpX.sum(v**2 + w**2, include_last=True))
ocpX.subject_to( (-u_max_speed <= v) <= u_max_speed )
ocpX.subject_to( (-u_max_rotation <= w) <= u_max_rotation )

# Initial condition
X_0 = ocpX.register_parameter(MX.sym('X_0', nx))
ocpX.subject_to(ocpX.at_t0(Xx)==X_0)

# Parameters for copies and multipliers
lambda_i = ocpX.register_parameter(MX.sym('lambda_i', nz), grid='control', include_last=True)
copy_i = ocpX.register_parameter(MX.sym('copy_i', nz), grid='control', include_last=True)
lambda_ji = ocpX.register_parameter(MX.sym('lambda_ji', nz*number_of_robots), grid='control', include_last=True)
copy_ji = ocpX.register_parameter(MX.sym('copy_ji', nz*number_of_robots), grid='control', include_last=True)

c_i = copy_i - X
term_i = dot(lambda_i, c_i) + mu/2*sumsqr(c_i)
if ocpX.is_signal(term_i):
    term_i = ocpX.sum(term_i,include_last=True)
ocpX.add_objective(term_i)

for j in range(number_of_robots):
    copy_j = vertcat(copy_ji[2*j], copy_ji[2*j+1])
    lambda_j = vertcat(lambda_ji[2*j], lambda_ji[2*j+1])
    c_j = copy_j - X
    term_j = dot(lambda_j, c_j) + mu/2*sumsqr(c_j)
    if ocpX.is_signal(term_j):
        term_j = ocpX.sum(term_j,include_last=True)
    ocpX.add_objective(term_j)

options = {"ipopt": {
    "print_level": 3,
    # "linear_solver": "ma27",
    # "tol": 1e-6,
    'sb': 'yes',
    }}
options["expand"] = True
options["print_time"] = False
options['record_time'] = True
ocpX.solver('ipopt',options)
ocpX.method(MultipleShooting(N=Nhor,M=1,intg='rk'))

#ocpX._method.add_sampler("u", u)
#ocpX._method.add_sampler("v", v)

## Set dummies for parameter values (required for ocp.to_function)
## In case you create the OCP function without solving the OCP first
ocpX.set_value(x_ref, 0)
ocpX.set_value(y_ref, 0)

ocpX.set_value(lambda_i, DM.zeros(nz, Nhor+1))
ocpX.set_value(copy_i, DM.zeros(nz, Nhor+1))

ocpX.set_value(lambda_ji, DM.zeros(nz*number_of_robots, Nhor+1))
ocpX.set_value(copy_ji, DM.zeros(nz*number_of_robots, Nhor+1))

ocpX.set_value(X_0, [0, 0, 0])

#####################################
## Define values for testing
#####################################
lambda_value = np.zeros([nz, Nhor+1])
copy_value = np.zeros([nz, Nhor+1])
lambda_values = np.zeros([nz*number_of_robots, Nhor+1])
copy_values = np.zeros([nz*number_of_robots, Nhor+1])

#####################################
## Solve OCP with from OCP object
#####################################
## Set parameters
ocpX.set_value(x_ref, 1.0)
ocpX.set_value(y_ref, 1.0)

ocpX.set_value(lambda_i, lambda_value)
ocpX.set_value(copy_i, copy_value)

ocpX.set_value(lambda_ji, lambda_values)
ocpX.set_value(copy_ji, copy_values)

ocpX.set_value(X_0, [0, 0, 0])

## Execute the solver
ocpX.solve()

#########################################
## Create CasADi function from OCP object
#########################################
## Sample variables to get their symbolic representation
xref_samp = ocpX.value(x_ref)
yref_samp = ocpX.value(y_ref)

li_samp = ocpX.sample(lambda_i, grid='control')[1]
ci_samp = ocpX.sample(copy_i, grid='control')[1]
lji_samp = ocpX.sample(lambda_ji, grid='control')[1]
cji_samp = ocpX.sample(copy_ji, grid='control')[1]

X_0_samp = ocpX.value(X_0)

_, x_samp = ocpX.sample(x, grid='control')
_, y_samp = ocpX.sample(y, grid='control')
_, v_samp = ocpX.sample(v, grid='control-')
_, w_samp = ocpX.sample(w, grid='control-')

## Define inputs and outputs of CasADi function
inputs = [xref_samp, yref_samp, li_samp, ci_samp, lji_samp, cji_samp, X_0_samp]
input_names = ['xref_samp', 'yref_samp', 'li_samp', 'ci_samp', 'lji_samp', 'cji_samp', 'X_0_samp']

outputs = [x_samp, y_samp, v_samp, w_samp]
output_names = ['x_samp', 'y_samp', 'v_samp', 'w_samp']

## Create CasADi function
ocpX_function = ocpX.to_function('u_ocpX', 
                                inputs,
                                outputs, 
                                input_names,
                                output_names)

## Serialize function
ocpX_function.save('u_ocpX.casadi')

## Test
ocpX_function(1.0, 0.0, lambda_value, copy_value, lambda_values, copy_values, [0, 0, 0])


######################################
# Get discrate dynamics for simulation
######################################
# Dynamics declaration
#Sim_asv_dyn = ocpX._method.discrete_system(ocpX)

# Sim_asv_dyn.save('sim_asv_dyn.casadi')