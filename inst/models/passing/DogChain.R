library(OpenMx)

# nudgeZeroStarts will ensure that the inequality constraint is violated at the starting values
m1 <- mxModel("dogChain",
              mxMatrix(name="link", nrow=4, ncol=1, free=TRUE, lbound=-1, values=0),
              mxMatrix(name="dog", nrow=1, ncol=1, free=TRUE, values=0),
              mxFitFunctionAlgebra("dog"),
              mxConstraint(dog > link[1,1] + link[2,1] + link[3,1] + link[4,1]))
m1 <- mxRun(m1)
omxCheckCloseEnough(m1$dog$values, -4, 1e-4)
omxCheckCloseEnough(m1$link$values[,1], rep(-1, 4), 1e-4) 
omxCheckCloseEnough(m1$output$evaluations, 0, 200)
