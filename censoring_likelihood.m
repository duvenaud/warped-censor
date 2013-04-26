function lp = censoring_likelihood( X, Y, log_hypers, censoring_func, num_samples)
% Estimates the probability that a datapoint will be censored,
% given that it is drawn from a spherical Gaussian, warped by a GP-LVM,
% and then censored with probability given by censoring_func
%
% David Duvenaud
% April 2013

% Todo: Make it return gradients w.r.t. the censoring functions.

[N, D] = size(Y);
[latent_dimension] = size(X, 2);

% First, generate samples from the latent distribution.
latent_draws = randn(num_samples, latent_dimension);

% Compute Gram gram matrix.
hyp(1) = -log_hypers.gamma/2;
hyp(2) = log_hypers.alpha/2;
K = covSEiso(hyp, X) + eye(N)*max(exp(log_hypers.betainv), 1e-3);  % Add a little noise.

% Compute conditional posterior.
crosscov = covSEiso(hyp, latent_draws, X);
post_mean = crosscov*(K\Y);
prior_var = covSEiso(hyp, latent_draws, 'diag');
post_var = prior_var - sum(bsxfun(@times, crosscov/K, crosscov), 2);

observed_samples = post_mean + randn(num_samples, D) .* repmat(post_var, 1, D);

% Now evaluate the probability of censoring those
censoring_probs = censoring_func( observed_samples );

% Correction factor = num_samples / num_uncensored
lp = log(N) - log(sum(1 - censoring_probs));

 %N/ (sum(1 - censoring_probs))
