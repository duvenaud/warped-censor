function [ nll, dnll_X ] = latent_likelihood( X )
    % Computes the gradient of the latent points w.r.t. p(x),
    % where p(x) is a spherical Gaussian.
    
    [N, dim] = size(X);
    Sigma = eye(dim);
    logdetcov = logdet(Sigma);
    
    nll = (dim/2)*log(2*pi) + (.5)*logdetcov +...
           (.5.*sum(bsxfun( @times, X / Sigma, X), 2));   % Evaluate for multiple inputs.

    nll = sum(nll);
    
    dnll_X = X / Sigma;
end

function ld = logdet(K)
    % returns the log-determinant of posdef matrix K.
        ld = 2*sum(log(diag(chol(K))));

end 
