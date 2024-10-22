import Pkg

Pkg.activate(joinpath(@__DIR__, "."))
Pkg.update()
Pkg.resolve()
Pkg.instantiate()
