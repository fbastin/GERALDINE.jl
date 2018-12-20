mutable struct BFGS_Matrix <: AbstractMatrix{Float64}
    H::Matrix
end

function btr(f::Function, g!::Function, state::BTRState{BFGS_Matrix}, x0::Vector; 
        verbose::Bool = true, nmax::Int64 = 1000, epsilon::Float64 = 1e-4, 
        accumulate!::Function, accumulator::Array)
    n = length(x0)
    b = BTRDefaults()
    state.fx = f(x0)
    g!(x0, state.g)
    state.Δ = 0.1*norm(state.g)
    y = zeros(n)
    gcand = zeros(n)
    
    function model(s::Vector, g::Vector, H::Matrix)
        return dot(s, g)+0.5*dot(s, H*s)
    end
    
    while !Stop_optimize(state.fx, state.g, state.iter, nmax = nmax)
        accumulate!(state, accumulator)
        if verbose
            println(state)
        end
        state.step = TruncatedCG(state)
        state.xcand = state.x+state.step
        fcand = f(state.xcand)
        
        state.ρ = -(state.fx-fcand)/(dot(state.step, state.g)+0.5*dot(state.step, state.H*state.step))
        
        g!(state.xcand, gcand)
        y = gcand - state.g
        BFGS!(state.H, y, state.step)
        if acceptCandidate!(state, b)
            state.x = copy(state.xcand)
            state.g = copy(gcand)
            state.fx = fcand
        end
        updateRadius!(state, b)
        state.iter += 1
    end
    return state, accumulator
end

function BFGS!(bfgs::BFGS_Matrix, y::Vector, s::Vector)
    Bs = bfgs.H*s
    bfgs.H[:, :] += (y*y')/(y'*s) - (Bs*Bs')/(s'*Bs)
end



import Base.size
function size(a::BFGS_Matrix)
    return size(a.H)
end
import Base.getindex
function getindex(a::BFGS_Matrix, index...)
    return getindex(a.H, index...)
end
import Base.setindex!
function setindex!(a::BFGS_Matrix, value, index...)
    setindex!(a.H, value, index...)
end




function OPTIM_btr_BFGS(f::Function, g!::Function, x0::Vector; verbose::Bool = true, nmax::Int64 = 1000, epsilon::Float64)
    H = Array{Float64, 2}(I, length(x0), length(x0))
    
    function accumulate!(state::BTRState{BFGS_Matrix}, acc::Vector)
        push!(acc, state.fx)
    end
    accumulator = []
    state = BTRState(BFGS_Matrix(H))
    state.x = x0
    state.iter = 0
    state.g = zeros(length(x0))
    state, accumulator = btr(f, g!, state, x0,
        verbose = verbose, nmax = nmax, epsilon = epsilon, accumulate! = accumulate!, accumulator = accumulator)
    return state, accumulator
end
