import string

import sympy as sp


# Desired belt balancer values for generating the model
inbounds = sp.Integer(1)  # Number of inbound belts
outbounds = sp.Integer(3)  # Number of outbound belts
splitters = sp.Integer(3)  # Number of splitters allowed


# Derived values for inbound and outbound traffic
cargos = inbounds  # One unique homogeneous cargo type for each inbound belt
throughput = min(inbounds, outbounds)  # Number of belts of throughput
v_in = throughput / inbounds  # Total volume per inbound belt
v_out = throughput / outbounds  # Total volume per outbound belt
t_in = v_in / 1  # Inbound traffic per cargo per belt (homogeneous)
t_out = v_out / cargos  # Outbound traffic per cargo per belt


# Basic sets

# Inbound belts to the balancer
I = {f"i{i+1}" for i in range(inbounds)}

# Outbound belts from the balancer
J = {f"j{j+1}" for j in range(outbounds)}

# Splitters within the balancer
S = {f"s{s+1}" for s in range(splitters)}

# Unique cargo types for each inbound belt
C = {f"c{c+1}" for c in range(cargos)}


# Composite sets

# Possible belt starting points that can yield cargo
P = I | S  # union

# Possible belt ending points that can accept cargo
Q = J | S  # union

# Available belt routes within the balancer
B = {(p, q) for p in P for q in Q}

# Cross product of inbound belts and cargo types
IxC = {(i, c) for i in I for c in C}

# Cross product of outbound belts and cargo types
JxC = {(j, c) for j in J for c in C}

# Cross product of splitters and cargo types
SxC = {(s, c) for s in S for c in C}

# Cross product of internal belts and cargo types
BxC = {(b[0], b[1], c) for b in B for c in C}

# Cross product of splitter output belts and cargo types
SxQxC = {(s, q, c) for s in S for q in Q for c in C} & BxC


# Parameters

# Homogeneous cargo traffic for each inbound belt
def f_init(i, c):
    # Extract the int from each string and compare
    n_i = int(i.lstrip(string.ascii_letters))
    n_c = int(c.lstrip(string.ascii_letters))
    if n_i == n_c:
        return t_in
    return sp.Integer(0)
f = {(i, c): f_init(i, c) for i, c in IxC}
for value in f.values():
    assert 0 <= value <= 1

# Perfectly mixed cargo traffic for each outbound belt
g = {(j, c): t_out for j, c in JxC}
for value in g.values():
    assert 0 <= value <= 1

# Upper bound on the traffic flowing along any belt
u_t = sp.Integer(1)
assert u_t > 0

# Upper bound on the traffic flowing into any splitter
u_x = sp.Integer(2)
assert u_x > 0


# Variables

system = []
variables = []

# Decision variable indicating whether a belt exists
e = {
    (p, q): sp.Symbol(f"e_{p}_{q}", integer=True, nonnegative=True)
    for p, q in B
}
system.extend((var <= 1) for var in e.values())
variables.extend(e.values())

# Fraction of belt capacity occupied by one cargo type
t = {
    (p, q, c): sp.Symbol(f"t_{p}_{q}_{c}", nonnegative=True)
    for p, q, c in BxC
}
system.extend((var <= 1) for var in t.values())
variables.extend(t.values())

# Total occupied fraction of belt capacity
# Volume definition constraint
# Define belt volume as the sum over all cargo
v = {(p, q): sum(t[p, q, c] for c in C) for p, q in B}
system.extend((var <= 1) for var in v.values())

# Number of belts running into each splitter
# Constraint defining number of splitter input belts
# Define the number of input belts to a splitter with a sum
n = {s: sum(e[p, s] for p in P) for s in S}

# Number of belts running out of each splitter
# Constraint defining number of splitter output belts
# Define the number of output belts to a splitter with a sum
m = {s: sum(e[s, q] for q in Q) for s in S}

# Sum of traffic flowing into each splitter by cargo
# Constraint defining splitter inflow traffic by cargo
# Define splitter inflow as the sum over all input belts
x = {(s, c): sum(t[p, s, c] for p in P) for s, c in SxC}

# Sum of traffic flowing out of each splitter by cargo
# Constraint defining splitter outflow traffic by cargo
# Define splitter outflow as the sum over all output belts
y = {(s, c): sum(t[s, q, c] for q in Q) for s, c in SxC}

# Indicator variable for splitters that have two outputs
z = {s: sp.Symbol(f"z_{s}", integer=True, nonnegative=True) for s in S}
system.extend((var <= 1) for var in z.values())
variables.extend(z.values())


# Inbound/outbound connectedness constraints

# Balancer must connect all inbound belts exactly once
system.extend(sp.Eq(sum(e[i, q] for q in Q), 1) for i in I)

# Balancer must connect all outbound belts exactly once
system.extend(sp.Eq(sum(e[p, j] for p in P), 1) for j in J)


# Inbound/outbound balanced traffic constraints

# Balancer must consume all inbound traffic
system.extend(sp.Eq(sum(t[i, q, c] for q in Q), f[i, c]) for i, c in IxC)

# Balancer must produce expected outbound traffic
system.extend(sp.Eq(sum(t[p, j, c] for p in P), g[j, c]) for j, c in JxC)


# Belt capacity/connectedness constraint
# Ensure belt volume is limited by capacity and connectedness
system.extend((v[p, q] <= e[p, q]) for p, q in B)


# Splitter cargo conservation constraint
# Splitters do not create cargo or destroy cargo
system.extend(sp.Eq(x[s, c], y[s, c]) for s, c in SxC)


# Splitter connectedness constraints

# Splitters with two output belts should have at least one input belt
system.extend((n[s] >= z[s]) for s in S)

# Splitters can take at most two input belts
system.extend((n[s] <= 2) for s in S)

# Splitters that are actually splitting must have two output belts
system.extend((m[s] >= 2 * z[s]) for s in S)

# Splitters have one output belt unless they are actually splitting
system.extend((m[s] <= 1 + z[s]) for s in S)


# Constraint so that splitters split evenly
# By far the most complex constraint here (because it's not actually linear)
# Supposed to be equivalent to t[s, q, c] == e[s, q] * x[s, c] / (1 + z[s])
# Splitters split their total input evenly to their outputs
system.extend(
    sp.Eq(t[s, q, c], e[s, q] * x[s, c] / (1 + z[s]))
    for s, q, c in SxQxC)



sset = sp.nsolve(system, variables)


print("pause")
