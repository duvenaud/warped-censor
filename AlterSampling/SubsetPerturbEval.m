function p = SubsetPerturbEval (Xnew, Xold, perturbStd)
if (~exist('perturbStd', 'var'))
    perturbStd = 0.1;
end

assert(sum(size(Xnew) ~= size(Xold)) == 0);
N = size(Xnew, 1);
D = size(Xnew, 2);

warning off;
p = (1 ./ nchoosek(N, floor(N/10))) * prod(mvnpdf(Xnew(Xnew ~= Xold, :), Xold(Xnew ~= Xold, :), eye(D) * perturbStd^2));
warning on;

end