function Xold = SubsetPerturbSamp(Xold, perturbStd)

F = 3;

if (~exist('perturbStd', 'var'))
    perturbStd = 0.1;
end

N = size(Xold, 1);
D = size(Xold, 2);
subset = randperm(N, floor(N / F));

Xold(subset, :) = Xold(subset, :) + perturbStd * randn(floor(N / F), D);
end