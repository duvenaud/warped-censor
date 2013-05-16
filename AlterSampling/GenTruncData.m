%##########################################################################
% GenTruncData.m
% Generate truncated data.
%##########################################################################

%##########################################################################
% Setup the dataset
% But only do it if there isn't already a dataset defined...
%##########################################################################
% Dataset parameters
outD = 3;                       % GPLVM output dimension
latD = 1;                       % GPLVM latent space dimension
noiseLevel = 0.0009;            % Noise in output space
covfunc = @covSEiso;            % Covariance function in latent space
hyp.cov = [0 0];                % Hyperparameters
hyp.lik = 0.5*log(noiseLevel);  % Noiselevel in observations,
% convention as in GPML (log of stddev)

% Generate data and initialise
[Y X] = gplvmSamplePrior(200, outD, latD, noiseLevel, covfunc, hyp);
N = size(Y, 1);

% Define truncation function
pTruncEval = @(y) ( 10^-5 + ((y(:, 1) < 0) & (y(:, 2) < 0)) * 0.995 );
pTruncSamp = @(y) ( pTruncEval(y) > rand(size(y, 1), 1) );

truncateObservation = pTruncSamp(Y);

% Truncate the data
Yo     = Y(~truncateObservation, :);
No     = sum(~truncateObservation);
XoTrue = X(~truncateObservation, :);

YcTrue = Y(truncateObservation, :);
XcTrue = X(truncateObservation, :);
NcTrue = sum(truncateObservation);

%% ########################################################################
% Visualise dataset
%##########################################################################
fprintf('Censored points: %f%%\n', NcTrue / N * 100);

figure(1);
plot3(Yo(:, 1), Yo(:, 2), Yo(:, 3), 'x', YcTrue(:, 1), YcTrue(:, 2), YcTrue(:, 3), 'o');
tilefigs;