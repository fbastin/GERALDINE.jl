"""
G: Groupes

E: Elementary

R: Reaserch

A: Algorithm

L: Linear

D: Diploma

I: Information

N: Non Linear

E: Estimated


"""
module GERALDINE

using LinearAlgebra, Statistics


export OPTIM_AGRESSIVE_AG, OPTIM_BFGS, OPTIM_btr_TH, OPTIM_btr_BFGS, btr_BFGS, OPTIM_AGRESSIVE_RSAG, stop_stochastic_1

include("stop.jl")
include("AG.jl")
include("BFGS.jl")
include("BTR/BTRBASE.jl")
include("BTR/BTR_True_Hessian.jl")
include("BTR/BTR_BFGS.jl")
include("STO/stop_stoc.jl")
include("STO/RSAG.jl")

end # module
