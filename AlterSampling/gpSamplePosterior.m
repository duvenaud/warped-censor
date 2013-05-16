function [Yc KXo KXoXc KXcXc] = gpSamplePosterior(Yo, Xo, Xc, covfunc, hyp, KXo, KXoXc, KXcXc)

noiseVar = exp(2 * hyp.lik);
NumObserved = size(Xo, 1);

if ~exist('KXo', 'var')
    KXo = feval(covfunc, hyp.cov, Xo) + noiseVar * eye(NumObserved);
end
if ~exist('KXoXc', 'var')
    KXoXc = feval(covfunc, hyp.cov, Xo, Xc);
end
if ~exist('KXcXc', 'var')
    KXcXc = feval(covfunc, hyp.cov, Xc, Xc);
end

mean = KXoXc' / KXo * Yo;
cov = KXcXc - KXoXc'/KXo*KXoXc;

R = chol(cov + noiseVar * eye(size(cov)));
Yc = mean + R' * randn(size(cov, 1), size(Yo, 2));

end