mutable struct BFGS_Matrix <: AbstractMatrix{Float64}
    H::Matrix
end

function btr(f::Function, g!::Function, state::BTRState{BFGS_Matrix}; 
        verbose::Bool = true, nmax::Int64 = 1000, epsilon::Float64, 
        accumulate!::Function = (state, acc) -> nothing, accumulator = [])
    
    b = BTRDefaults()
    state.fx = f(x0)
    g!(x0, state.g)
    state.Δ = 0.1*norm(state.g)
    y = zeros(n)
    gcand = zeros(n)
    
    function model(s::Vector, g::Vector, H::Matrix)
        return dot(s, g)+0.5*dot(s, H*s)
    end
    
    while !Stop_optimize(fx, state.g, state.iter, nmax = nmax)
        accumulate!(accumulator)
        if verbose
            println(state)
        end
        state.step = TruncatedCG(state)
        state.xcand = state.x+state.step
        fcand = f(state.xcand)
        
        state.ρ = -(fx-fcand)/(dot(state.step, state.g)+0.5*dot(state.step, state.H*state.step))
        
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

function BFGS!(H::Matrix, y::Vector, s::Vector)
    Bs = H*s
    H[:, :] += (y*y')/(y'*s) - (Bs*Bs')/(s'*Bs)
end



import Base.size
function size(a::BFGS_Matrix)
    return size(a.a)
end
import Base.getindex
function getindex(a::BFGS_Matrix, index...)
    return getindex(a.a, index...)
end
import Base.setindex!
function setindex!(a::BFGS_Matrix, value, index...)
    setindex!(a.a, value, index...)
end




function OPTIM_btr_BFGS(f::Function, g!::Function, x0::Vector; verbose::Bool = true, nmax::Int64 = 1000, epsilon::Float64)
    H = BFGS_Matrix(Array{Float64, 2}(I, length(x0), length(x0)))
    state = BTRState(H)
    state.x = x0
    state.iter = 0
    state.g = zeros(length(x0))
    state = btr(f, g!, state,
        verbose = verbose, nmax = nmax, epsilon = epsilon)
    return x
end