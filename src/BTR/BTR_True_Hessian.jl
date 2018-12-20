function btr(f::Function, g!::Function, H!::Function, state::BTRState{Array{Float64,2}}, x0::Vector; 
        verbose::Bool = true, nmax::Int64 = 1000, epsilon::Float64 = 1e-6, 
        accumulate!::Function, accumulator::Array)
    b = BTRDefaults()
    state.fx = f(x0)
    g!(x0, state.g)
    state.Δ = 0.1*norm(state.g)
    H!(x0, state.H)
    

    function model(s::Vector, g::Vector, H::Matrix)
        return dot(s, g)+0.5*dot(s, H*s)
    end
    
    while !Stop_optimize(state.fx, state.g, state.iter, nmax = nmax, tol = epsilon)
        accumulate!(accumulator)
        if verbose
            println(state)
        end
        
        state.step = TruncatedCG(state, H)
        state.xcand = state.x+state.step
        fcand = f(state.xcand)
        state.ρ = (fcand-state.fx)/(dot(state.s, state.g)+0.5*dot(state.s, state.H*state.s))
        if acceptCandidate!(state, b)
            state.x = copy(state.xcand)
            g!(state.x, state.g)
            H!(state.x, state.H)
            
            state.fx = fcand
        end
        updateRadius!(state, b)
        state.iter += 1
    end
    return state, accumulator
end

function OPTIM_btr_TH(f::Function, g!::Function, H!::Function, 
                x0::Vector; verbose::Bool = true, 
                nmax::Int64 = 1000, epsilon::Float64 = 1e-4)
        
        
    function accumulate!(state::BTRState{Matrix}, acc::Vector)
        push!(acc, state.fx)
    end
    accumulator = []
        
    H = Array{Float64, 2}(I, length(x0), length(x0))
    state = BTRState(H)
    println(typeof(state))
    state.x = copy(x0)
    state.iter = 0
    state.g = zeros(length(x0))
        
    
    state, accumulator = btr(f, g!, H!, state, x0, 
                verbose = verbose, nmax = nmax, epsilon = epsilon, accumulate! = accumulate!, accumulator)
    return state, accumulator
end
