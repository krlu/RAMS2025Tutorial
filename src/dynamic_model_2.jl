module DynamicModel2

using Base: Float64
using Scruff
using Scruff.Utils
using Scruff.RTUtils
using Scruff.Models
using Scruff.SFuncs
using Scruff.Algorithms
import Scruff: make_initial, make_transition

struct MaintenanceModel <: VariableTimeModel{Tuple{}, Tuple{Bool}, Bool}
end

make_initial(::MaintenanceModel, t) = Cat([true, false], [0.5, 0.5])
make_transition(::MaintenanceModel, parts, t) = 
    Chain(Tuple{Bool}, Bool, tuple -> begin 
        needsMaintenance = tuple[1]
        Mixture(
            [Constant(needsMaintenance), Cat([true, false], [0.5, 0.5])],
            [0.9, 0.1]
        )
    end)

struct TemperatureModel <: VariableTimeModel{Tuple{}, Tuple{Bool}, Float64}
end
make_initial(::TemperatureModel, t) = Normal(207.5, 4.0)
make_transition(::TemperatureModel, parts, t) = 
    Chain(Tuple{Bool}, Float64, tuple -> begin 
        nmBool = tuple[1]
        # no dependence on previous temperature results in unstable temperature inference
        if nmBool 
            Uniform(0.0, 300.0)
        else
            Normal(207.5, 4.0)
        end
    end)

function run_inference()
    needsMaintenance = MaintenanceModel()(:needsMaintenance)
    temperature = TemperatureModel()(:temp)
    variables = [needsMaintenance, temperature]
    graph = VariableGraph(
        temperature => [needsMaintenance], 
        needsMaintenance => [needsMaintenance]
    )
    net = DynamicNetwork(variables, VariableGraph(), graph)
    runtime = Runtime(net)
    numParticles = 10000
    pf = AsyncPF(numParticles, numParticles, Int)
    init_filter(pf, runtime)

    score = HardScore{Float64}(207.5)
    for t in 1:50
        evidence = 
            if t == 1 
                Dict{Symbol, Score}(:temp => score)
            else
                Dict{Symbol, Score}()
            end
        filter_step(pf, runtime, [needsMaintenance, temperature], t, evidence)
        sampledResults = get_state(runtime, :particles)
        probNeedsMaintenance = probability(sampledResults, x -> begin 
            x[Symbol("needsMaintenance_$t")]
        end)
        # println("results at time $t")
        # println(probTemp)
        println(probNeedsMaintenance)

    end
end 

run_inference()

end