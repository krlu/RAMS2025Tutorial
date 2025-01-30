module StaticModel

using Base: Float64
using Scruff
using Scruff.Utils
using Scruff.RTUtils
using Scruff.Models
using Scruff.SFuncs
using Scruff.Algorithms

needsMaintenanceSF = Cat([true, false], [0.5, 0.5])

tempSF = Chain(Tuple{Bool}, Float64, tuple -> begin 
    nmBool = tuple[1]
    if nmBool 
        Uniform(0.0, 300.0)
    else
        Normal(207.5, 4.0)
    end
end)

nmModel = SimpleModel(needsMaintenanceSF)
needsMaintenance = nmModel(:needsMaintenance)

tempModel = SimpleModel(tempSF)
temperature = tempModel(:temp)

variables = [temperature, needsMaintenance]
graph = VariableGraph(temperature => [needsMaintenance])
net = InstantNetwork(variables, graph)
runtime = Runtime(net)
numSamples = 10000
alg = LW(numSamples)

observedTemp = 207.5
score = HardScore(observedTemp)
evidence = Dict{Symbol, Score}(:temp => score)
sampledResults = infer(alg, runtime, evidence)

probNeedsMaintenance = probability(sampledResults, x -> begin 
    x[:needsMaintenance]
end)

probTemp = probability(sampledResults, x -> begin 
    199 < x[:temp] < 201
end)


println(probTemp)
println(probNeedsMaintenance)

end