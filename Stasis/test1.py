from diffeqpy import de
import PRyM.PRyM_jl_sys as PRyMjl
from julia import Main

Main.eval("PRyMjl.dlnajl([0.0], [0.0], [0.0], 0.0)")

def my_ode(du, u, p, t):
    du[0] = -0.5 * u[0]
    du[1] = 0.5 * (u[0] - u[1])

u0 = [1.0, 0.0]
tspan = (0.0, 10.0)
prob = de.ODEProblem(PRyMjl.dlnajl, u0, tspan)
sol = de.solve(prob, de.Tsit5())

print("Solution at time points:", sol.t)
print("Solution values:", sol.u)
