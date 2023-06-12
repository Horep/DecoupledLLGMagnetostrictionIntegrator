# DecoupledLLGMagnetostrictionIntegrator
This uses NGSolve/Netgen to solve the Landau-Lifshitz-Gilbert equation including magnetostriction, coupled with the conservation of momentum equation.

We use a tangent plane scheme without a projection step.

Should be functional, but no guarantees.

# Important information:

Written for Python 3.7.9. Future version may work, but no guarantees, and you are likely to get different behaviour in some areas (e.g. CGSolver will spit out loads of stuff into your console during solving.)

See requirements.txt for the required imports, and the versions I used to make the code.
