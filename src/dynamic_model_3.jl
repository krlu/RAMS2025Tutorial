module DynamicModel3

using Base: Float64
using Scruff
using Scruff.Utils
using Scruff.RTUtils
using Scruff.Models
using Scruff.SFuncs
using Scruff.Algorithms
import Scruff: make_initial, make_transition

# add simple change in temperature 
# show prognostics 
# tweak slide languages for example 3 to reflect this 

temp_mean = 207.5
temp_var = 4.0

struct MaintenanceModel <: VariableTimeModel{Tuple{}, Tuple{Float64}, Bool}
end
make_initial(::MaintenanceModel, t) = Cat([true, false], [0.5, 0.5])
make_transition(::MaintenanceModel, parts, t) = 
    Chain(Tuple{Float64}, Bool, tuple -> begin 
        temp = tuple[1]
        if temp - temp_mean > temp_var
            Constant(true)
        else
            Constant(false)
        end
    end)

struct TemperatureModel <: VariableTimeModel{Tuple{}, Tuple{Float64}, Float64}
end
make_initial(::TemperatureModel, t) = Normal(temp_mean, temp_var)
make_transition(::TemperatureModel, parts, t) = 
    Chain(Tuple{Float64}, Float64, tuple -> begin 
        prevTemp = tuple[1] + 1
        Normal(prevTemp, temp_var)
    end)

function run_inference()
    needsMaintenance = MaintenanceModel()(:needsMaintenance)
    temperature = TemperatureModel()(:temp)
    variables = [needsMaintenance, temperature]
    graph = VariableGraph(
        temperature => [temperature], 
        needsMaintenance => [temperature]
    )
    net = DynamicNetwork(variables, VariableGraph(), graph)
    runtime = Runtime(net)
    numParticles = 10000
    pf = AsyncPF(numParticles, numParticles, Int)
    init_filter(pf, runtime)

    observedTemp = 207.5
    for t in 1:50
        score = HardScore(observedTemp)
        evidence = 
            if t == 1
                Dict{Symbol, Score}(:temp => score)
            else
                Dict{Symbol, Score}()
            end
        filter_step(pf, runtime, [needsMaintenance, temperature], t, evidence)
        sampledResults = get_state(runtime, :particles)
        
        probTemp= probability(sampledResults, x -> begin 
            abs(x[Symbol("temp_$t")] - observedTemp) < 2.0
        end)
        # println("results at time $t")
#         println(probTemp)
        probNeedsMaintenance = probability(sampledResults, x -> begin 
            x[Symbol("needsMaintenance_$t")]
        end)
        println(probNeedsMaintenance)

    end
end 

run_inference()

end