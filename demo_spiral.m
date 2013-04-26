function demo_spiral()
% A simple demo of the warped mixture model.
%
% David Duvenaud
%
% April 2013
% ====================

addpath('utils');
addpath('data');
addpath('gpml/cov');
addpath('gpml/util');

% Set the random seed.
seed = 0;
randn('state', seed);
rand('twister', seed);    

% Load the dataset.
dataset = 'spiral2';
fn = sprintf('data/%s.mat', dataset);
load(fn);      % Load X (observed data) and y (true cluster labels)

% Set some options.
options = [];
options.GPLVMinit = 0;   % Initialize using GP-LVM.

% HMC sampler options. 
% Adjust these until you get a mix of accepts and rejects.
options.hmc.epsilon = 0.02;    % Step size.
options.hmc.tau = 25;          % Number of steps.
options.hmc.plot = false;

% Plotting options.
options.plot_period = 100;      % How often to plot.
options.record_frames = false; % Whether to record frames.
options.hmc_plot = true;       % Whether to plot hmc paths.
options.num_iters = 10000;

%figure('Position',[100 200 1200 1000]); clf;
        
latent_dimension = 2;

% Now call inference, with plotting turned on.
[sampled_latents, nlls] = gplvm_run_hmc(X, latent_dimension, options);


