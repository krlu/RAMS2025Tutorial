using Base: Float64
using Scruff
using Scruff.Utils
using Scruff.RTUtils
using Scruff.Models
using Scruff.SFuncs
using Scruff.Algorithms
import Scruff: make_initial, make_transition

needsMaintenanceSF = Cat([true, false], [0.5, 0.5])
tempSF = Chain(Tuple{Bool}, Float64, tuple -> begin 
    nmBool = tuple[1]
    if nmBool 
        Uniform(0.0, 300.0)
    else
        Normal(207.5, 4.0) # taken from domain expertise
    end
end)

nmModel = SimpleModel(needsMaintenanceSF)
nmVar = nmModel(:needsMaintenance)

tempModel = SimpleModel(tempSF)
tempVar = tempModel(:temp)

variables = [nmVar, tempVar]
graph = VariableGraph(tempVar => [nmVar])
net = InstantNetwork(variables, graph)
runtime = Runtime(net)
alg = LW(1000)

observedTemp = 195.0
score = FunctionalScore{Float64}(temp -> (begin
    1.0/abs(temp - observedTemp)^2
end))
parts = infer(alg, runtime, Dict{Symbol, Score}(:temp => score))
prob = probability(parts, x -> begin 
    x[:needsMaintenance]
end)
println(prob)