function [Yc R] = gpSamplePrior(Xo, covfunc, hyp, OutNum, R)

noiseVar = exp(2 * hyp.lik);
NumObserved = size(Xo, 1);

if ~exist('OutNum', 'var')
    OutNum = 1;
end

if ~exist('R', 'var')
    KXo = feval(covfunc, hyp.cov, Xo) + noiseVar * eye(NumObserved);
    R = chol(KXo);
end

Yc = R' * randn(size(Xo, 1), OutNum);

end