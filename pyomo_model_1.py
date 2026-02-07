import string

import highspy
import pyomo.environ as pyo
import pyomo.opt as opt


assert highspy is not None


# Desired belt balancer values for generating the model
inbounds = 8  # Number of inbound belts
outbounds = 8  # Number of outbound belts
splitters = 10  # Number of splitters allowed


# Derived values for inbound and outbound traffic
cargos = inbounds  # One unique homogeneous cargo type for each inbound belt
throughput = min(inbounds, outbounds)  # Number of belts of throughput
v_in = throughput / inbounds  # Total volume per inbound belt
v_out = throughput / outbounds  # Total volume per outbound belt
t_in = v_in / 1  # Inbound traffic per cargo per belt (homogeneous)
t_out = v_out / cargos  # Outbound traffic per cargo per belt


model = pyo.ConcreteModel()


# Functions for set utility operations
def init_set(letter, size):
    return [f"{letter}{x+1}" for x in range(size)]
def get_letter(member):
    return member.rstrip(string.digits)
def get_int(member):
    return int(member.lstrip(string.ascii_letters))


# Basic sets
model.I = pyo.Set(
    doc="Inbound belts to the balancer",
    initialize=init_set('i', inbounds),
)
model.J = pyo.Set(
    doc="Outbound belts from the balancer",
    initialize=init_set('j', outbounds),
)
model.S = pyo.Set(
    doc="Splitters within the balancer",
    initialize=init_set('s', splitters),
)
model.C = pyo.Set(
    doc="A unique cargo type for each inbound belt",
    initialize=init_set('c', cargos),
)


# Composite sets
model.P = pyo.Set(
    doc="Every possible producing end of a belt",
    initialize=model.I | model.S,  # Union
)
model.Q = pyo.Set(
    doc="Every possible consuming end of a belt",
    initialize=model.S | model.J,  # Union
)
def B_filter(model, p, q):
    if get_letter(p) == 'i' and get_letter(q) == 'j':
        return False  # Don't allow belts that bypass the splitters entirely
    if get_letter(p) == get_letter(q) == 's' and get_int(p) == get_int(q):
        return False  # Don't allow a splitter to loop right back on itself
    return True
model.B = pyo.Set(
    doc="Available belt routes within the balancer",
    initialize=model.P * model.Q,  # Cartesian product
    filter=B_filter,
)
model.IxC = pyo.Set(
    doc="Cartesian product of inbound belts and cargo types",
    initialize=model.I * model.C,
)
model.JxC = pyo.Set(
    doc="Cartesian product of outbound belts and cargo types",
    initialize=model.J * model.C,
)
model.SxC = pyo.Set(
    doc="Cartesian product of splitters and cargo types",
    initialize=model.S * model.C,
)
model.BxC = pyo.Set(
    doc="Cartesian product of internal belts and cargo types",
    initialize=model.B * model.C,
)
model.W = pyo.Set(
    doc="Belts that start at the output of a splitter",
    initialize=(model.S * model.Q) & model.B,  # Intersection
    within=model.B,
)
model.WxC = pyo.Set(
    doc="Cartesian product of splitter output belts and cargo types",
    initialize=model.W * model.C,
    within=model.BxC,
)


# Parameters
def f_init(model, i, c):
    if get_int(i) == get_int(c):
        return t_in
    return 0
model.f = pyo.Param(
    model.IxC,
    domain=pyo.PercentFraction,
    initialize=f_init,
    doc="Homogeneous cargo traffic for each inbound belt",
)
model.g = pyo.Param(
    model.JxC,
    domain=pyo.PercentFraction,
    default=t_out,
    doc="Perfectly mixed cargo traffic for each outbound belt",
)
model.u_t = pyo.Param(
    domain=pyo.PositiveReals,
    default=1,
    doc="Upper bound on the traffic flowing along any belt",
)
model.u_x = pyo.Param(
    domain=pyo.PositiveReals,
    default=2,
    doc="Upper bound on the traffic flowing into any splitter",
)


# Variables
model.e = pyo.Var(
    model.B,
    domain=pyo.Binary,
    doc="Decision variable indicating whether a belt exists",
)
model.t = pyo.Var(
    model.BxC,
    domain=pyo.PercentFraction,
    doc="Fraction of belt capacity occupied by one cargo type",
)
model.v = pyo.Var(
    model.B,
    domain=pyo.PercentFraction,
    doc="Total occupied fraction of belt capacity",
)
model.n = pyo.Var(
    model.S,
    domain=pyo.NonNegativeIntegers,
    doc="Number of belts running into each splitter",
)
model.m = pyo.Var(
    model.S,
    domain=pyo.NonNegativeIntegers,
    doc="Number of belts running out of each splitter",
)
model.x = pyo.Var(
    model.SxC,
    domain=pyo.NonNegativeReals,
    doc="Sum of traffic flowing into each splitter by cargo",
)
model.y = pyo.Var(
    model.SxC,
    domain=pyo.NonNegativeReals,
    doc="Sum of traffic flowing out of each splitter by cargo",
)
model.z = pyo.Var(
    model.S,
    domain=pyo.Binary,
    doc="Indicator variable for splitters that have two outputs",
)
model.xe = pyo.Var(
    model.WxC,
    domain=pyo.NonNegativeReals,
    doc="Product of the variables x[s, c] and e[s, q]",
)
model.tz = pyo.Var(
    model.WxC,
    domain=pyo.PercentFraction,
    doc="Product of the variables t[s, q, c] and z[s]",
)


# Volume definition constraint
def define_volume_rule(model, p, q):
    return model.v[p, q] == sum(model.t[p, q, c] for c in model.C)
model.define_volume = pyo.Constraint(
    model.B,
    rule=define_volume_rule,
    doc="Define belt volume as the sum over all cargo",
)


# Constraints defining number of splitter input/output belts
def define_num_inputs_rule(model, s):
    return model.n[s] == sum(model.e[p, q] for p, q in model.B if q == s)
model.define_num_inputs = pyo.Constraint(
    model.S,
    rule=define_num_inputs_rule,
    doc="Define the number of input belts to a splitter with a sum",
)
def define_num_outputs_rule(model, s):
    return model.m[s] == sum(model.e[p, q] for p, q in model.B if p == s)
model.define_num_outputs = pyo.Constraint(
    model.S,
    rule=define_num_outputs_rule,
    doc="Define the number of output belts to a splitter with a sum",
)


# Constraints defining splitter inflow/outflow traffic by cargo
def define_inflow_rule(model, s, c):
    return model.x[s, c] == sum(model.t[p, q, c] for p, q in model.B if q == s)
model.define_inflow = pyo.Constraint(
    model.SxC,
    rule=define_inflow_rule,
    doc="Define splitter inflow as the sum over all input belts",
)
def define_outflow_rule(model, s, c):
    return model.y[s, c] == sum(model.t[p, q, c] for p, q in model.B if p == s)
model.define_outflow = pyo.Constraint(
    model.SxC,
    rule=define_outflow_rule,
    doc="Define splitter outflow as the sum over all output belts",
)


# Trick constraints to force xe[s, q, c] == x[s, c] * e[s, q]
# See section 7.7 of the AIMMS PDF in this directory for more details
def force_xe_1_rule(model, s, q, c):
    return model.xe[s, q, c] <= model.u_x * model.e[s, q]
model.force_xe_1 = pyo.Constraint(
    model.WxC,
    rule=force_xe_1_rule,
    doc="First inequality to force xe[s, q, c] == x[s, c] * e[s, q]",
)
def force_xe_2_rule(model, s, q, c):
    return model.xe[s, q, c] <= model.x[s, c]
model.force_xe_2 = pyo.Constraint(
    model.WxC,
    rule=force_xe_2_rule,
    doc="Second inequality to force xe[s, q, c] == x[s, c] * e[s, q]",
)
def force_xe_3_rule(model, s, q, c):
    return model.xe[s, q, c] >= model.x[s, c] - model.u_x * (1 - model.e[s, q])
model.force_xe_3 = pyo.Constraint(
    model.WxC,
    rule=force_xe_3_rule,
    doc="Third inequality to force xe[s, q, c] == x[s, c] * e[s, q]",
)


# Trick constraints to force tz[s, q, c] == t[s, q, c] * z[s]
# See section 7.7 of the AIMMS PDF in this directory for more details
def force_tz_1_rule(model, s, q, c):
    return model.tz[s, q, c] <= model.u_t * model.z[s]
model.force_tz_1 = pyo.Constraint(
    model.WxC,
    rule=force_tz_1_rule,
    doc="First inequality to force tz[s, q, c] == t[s, q, c] * z[s]",
)
def force_tz_2_rule(model, s, q, c):
    return model.tz[s, q, c] <= model.t[s, q, c]
model.force_tz_2 = pyo.Constraint(
    model.WxC,
    rule=force_tz_2_rule,
    doc="Second inequality to force tz[s, q, c] == t[s, q, c] * z[s]"
)
def force_tz_3_rule(model, s, q, c):
    return model.tz[s, q, c] >= model.t[s, q, c] - model.u_t * (1 - model.z[s])
model.force_tz_3 = pyo.Constraint(
    model.WxC,
    rule=force_tz_3_rule,
    doc="Third inequality to force tz[s, q, c] == t[s, q, c] * z[s]",
)


# Inbound/outbound connectedness constraints
def connect_inbounds_rule(model, i):
    return sum(model.e[p, q] for p, q in model.B if p == i) == 1
model.connect_inbounds = pyo.Constraint(
    model.I,
    rule=connect_inbounds_rule,
    doc="Balancer must connect all inbound belts exactly once",
)
def connect_outbounds_rule(model, j):
    return sum(model.e[p, q] for p, q in model.B if q == j) == 1
model.connect_outbounds = pyo.Constraint(
    model.J,
    rule=connect_outbounds_rule,
    doc="Balancer must connect all outbound belts exactly once",
)


# Inbound/outbound balanced traffic constraints
def consume_inbounds_rule(model, i, c):
    return sum(model.t[p, q, c] for p, q in model.B if p == i) == model.f[i, c]
model.consume_inbounds = pyo.Constraint(
    model.IxC,
    rule=consume_inbounds_rule,
    doc="Balancer must consume all inbound traffic",
)
def produce_outbounds_rule(model, j, c):
    return sum(model.t[p, q, c] for p, q in model.B if q == j) == model.g[j, c]
model.produce_outbounds = pyo.Constraint(
    model.JxC,
    rule=produce_outbounds_rule,
    doc="Balancer must produce expected outbound traffic",
)


# Belt capacity/connectedness constraint
def respect_capacity_rule(model, p, q):
    return model.v[p, q] <= model.e[p, q]
model.respect_capacity = pyo.Constraint(
    model.B,
    rule=respect_capacity_rule,
    doc="Ensure belt volume is limited by capacity and connectedness",
)


# Splitter cargo conservation constraint
def splitters_conserve_rule(model, s, c):
    return model.x[s, c] == model.y[s, c]
model.splitters_conserve = pyo.Constraint(
    model.SxC,
    rule=splitters_conserve_rule,
    doc="Splitters do not create cargo or destroy cargo",
)


# Splitter connectedness constraints
def min_inputs_rule(model, s):
    return model.n[s] >= model.z[s]
model.min_inputs = pyo.Constraint(
    model.S,
    rule=min_inputs_rule,
    doc="Splitters with two output belts should have at least one input belt",
)
def max_inputs_rule(model, s):
    return model.n[s] <= 2
model.max_inputs = pyo.Constraint(
    model.S,
    rule=max_inputs_rule,
    doc="Splitters can take at most two input belts",
)
def min_outputs_rule(model, s):
    return model.m[s] >= 2 * model.z[s]
model.min_outputs = pyo.Constraint(
    model.S,
    rule=min_outputs_rule,
    doc="Splitters that are actually splitting must have two output belts",
)
def max_outputs_rule(model, s):
    return model.m[s] <= 1 + model.z[s]
model.max_outputs = pyo.Constraint(
    model.S,
    rule=max_outputs_rule,
    doc="Splitters have one output belt unless they are actually splitting",
)


# Constraint so that splitters split evenly
# By far the most complex constraint here (because it's not actually linear)
# Supposed to be equivalent to t[s, q, c] == e[s, q] * x[s, c] / (1 + z[s])
def split_evenly_rule(model, s, q, c):
    return model.xe[s, q, c] == model.t[s, q, c] + model.tz[s, q, c]
model.split_evenly = pyo.Constraint(
    model.WxC,
    rule=split_evenly_rule,
    doc="Splitters split their total input evenly to their outputs",
)


# Objective
def simultaneous_volume_rule(model):
    return sum(model.v[p, q] for p, q in model.B)
model.simultaneous_volume = pyo.Objective(
    rule=simultaneous_volume_rule,
    sense=pyo.minimize,
    doc="Minimize sum of volumes across all belts simultaneously",
)


# Print entire prepared model
model.pprint()


# Solve the model
results = opt.SolverFactory("appsi_highs").solve(model)
results.write()


# Print the solution variables
print('', 60 * '=', 'BELT BALANCER SOLUTION', 60 * '=', '', sep='\n')
model.e.display()
model.t.display()
model.v.display()
model.n.display()
model.m.display()
model.x.display()
model.y.display()
model.z.display()
model.xe.display()
model.tz.display()

# Print the solution in a friendlier way
print()
print("Belt connections in solution:")
for p, q in model.B:
    if pyo.value(model.e[p, q]) > 0.5:
        v = pyo.value(model.v[p, q])
        print(f"    {p} -> {q}    v = {v:.8f}")
