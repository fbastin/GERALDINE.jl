function btr_TH(f::Function, g!::Function, H!::Function, x0::Vector; 
        state::BTRState = BTRState(), vervose::Bool = true, nmax::Int64 = 1000)
    b = BTRDefaults()
    state.iter = 0
    state.x = x0
    n = length(x0)
    tol2 = state.tol*state.tol
    state.g = zeros(n)
    H = Array{Float64, 2}(I, n, n)
    fx = f(x0)
    g!(x0, state.g)
    state.Δ = 0.1*norm(state.g)
    H!(x0, H)
    

    function model(s::Vector, g::Vector, H::Matrix)
        return dot(s, g)+0.5*dot(s, H*s)
    end
    
    while !Stop_optimize(fx, state.g, state.iter, nmax = nmax)
        
        if verbose
            println(round.(state.x, 2))
        end
        
        state.step = TruncatedCG(state, H)
        state.xcand = state.x+state.step
        fcand = f(state.xcand)
        state.ρ = (fcand-fx)/(model(state.step, state.g, H))
        if acceptCandidate!(state, b)
            state.x = copy(state.xcand)
            g!(state.x, state.g)
            H!(state.x, H)
            
            fx = fcand
        end
        updateRadius!(state, b)
        state.iter += 1
    end
    return state.x, state.iter
end

function OPTIM_btr_TH(f::Function, g!::Function, H!::Function, x0::Vector; verbose::Bool = true, nmax::Int64 = 1000)
    x, it = btr_TH(f, g!, H!, x0, vervose = verbose, nmax = nmax)
    return x
end