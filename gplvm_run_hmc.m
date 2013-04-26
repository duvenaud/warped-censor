function [sampled_latents, nlls] = gplvm_run_hmc(Y, latent_dimension, options)
% GP-LVM with Gaussians latent p(X).
%
% latent_dimension is the dimension of the latent space.
% Y is the observed data.
%
% This version generates samples from the posterior instead of 
% maximizing likelihood.  It works by alternatively sampling p(c|x) and p(x|c).
%
% Tomo and David
% April 2012


if nargin < 2;
    options = {};
    
end

[N,D] = size(Y);

% Set defaults.
if ~isfield( options, 'record_frames' )
    options.record_frames = false;
end
if ~isfield( options, 'plot_period' )
    options.plot_period = 0;
end
if ~isfield( options, 'num_iters' )
    options.num_iters = 100;
end
if isfield( options.hmc, 'epsilon' )
    options.hmc.epsilon = 0.02;
end
if isfield( options.hmc, 'tau' )
    options.hmc.tau = 25;
end
if isfield( options.hmc, 'plot' )
    options.hmc.plot = false;
end
if isfield( options, 'GPLVM_init' )
    options.GPLVM_init = false;
end

% Centering
%Y = Y-repmat(mean(Y,1),N,1);

% Initalize X as same as the observed data Y
if latent_dimension == D
    init_X = Y;
elseif options.GPLVM_init == 1
    gplvm_params = gplvm_original(Y,latent_dimension,[]);
    init_X = gplvm_params.X;
elseif latent_dimension <= D
    % Use first D dimensions.
    init_X = Y(:,1:latent_dimension);
else
    init_X(:,1:D) = Y;
    init_X(:,D+1:end) = 0;
end
if sum(sum(init_X)) == 0
    init_X = init_X + randn(N,latent_dimension);
end

% Initialize kernel hyperparameters.
% 'a (output variance)', '1/b (noise level)', '1/g (lengthscale squared)'
log_hypers.alpha = -1;
log_hypers.betainv = -1;
log_hypers.gamma = -1;



% Put all the parameters that will be optimized into a struct.
params.X = init_X;
params.log_hypers = log_hypers;

last_nll = NaN;

%censoring_func = @(x) zeros(size(x,1), 1);%x(:,1) > -2 & x(:,1) < 0;
censoring_func = @(x) x(:,1) > -2 & x(:,1) < 0 & x(:,2) < 2 & x(:, 2) > 1;

num_censoring_samples = 100;

% Main inference loop
for i = 1:options.num_iters

    % Convert structure of parameters into a vector.
    unwrapped_params = unwrap( params );

    %checkgrad('joint_likelihood', unwrapped_params, 1e-6, Y, params );
    [unwrapped_params, last_nll] = ea_hmc(@joint_likelihood, unwrapped_params, ...
                                          last_nll, options.hmc, Y, params, ...
                                          censoring_func, num_censoring_samples);
                                 
    nll(i) = last_nll;

    % Re-pack parameters.
    params = rewrap( params, unwrapped_params );

    
   % save GPLVM params
    run_hypers_alpha(i) = exp( params.log_hypers.alpha ) ;
    run_hypers_inv_beta(i) = exp( params.log_hypers.betainv ) ;
    run_hypers_gamma(i) = exp( params.log_hypers.gamma ) ;   

    %drawing
    if options.record_frames == 1
        draw_latent_representation( params.X, mix, assignments, labels );
        axis( [-7 5 -5 7]);
        drawnow;
        F(i) = getframe;
    end
    
    if options.plot_period ~= 0
        if mod(i,options.plot_period) == 0 || i == options.num_iters
            % draw_latent_representation
            figure(234234);
            if latent_dimension == 2
                clf; plot( params.X(:,1), params.X(:,2), '.' );
                title('Latent space');
            end

            if D == 2
                % Draw the original data and current assigments
                draw_warped_density( params.X, Y, params.log_hypers );
                plot_censoring_func( censoring_func );
            end
            
            % Sanity check
            figure(123); clf;
            plot( nll, 'b-' );
            ylabel('negative log likelihood');

            % Plot hypers over time
            figure(123423); clf;
            plot( run_hypers_alpha, 'b-' ); hold on;
            plot( run_hypers_inv_beta, 'r-' ); hold on;
            plot( 1./run_hypers_gamma, 'g-' ); hold on;
            legend({'a (output variance)', '1/b (noise level)', '1/g (lengthscale squared)'});
            title('hyperparams');
            
            drawnow;
        end
    end
end

end



function plot_censoring_func( censoring_func )
    x_lims = xlim;
    y_lims = ylim;
    N_1d = 100;
    
    xrange = linspace( x_lims(1), x_lims(2), N_1d);   % Choose a set of x locations.
    yrange = linspace( y_lims(1), y_lims(2), N_1d);   % Choose a set of x locations.
    [xvals, yvals] = meshgrid( xrange, yrange);
    gridvals = [xvals(:) yvals(:)];
    %figure(1); clf;
    
    probs1 = censoring_func(gridvals);
    
    % Probability of being in class 1.
    %probs1 = exp(trainscored(:,1))./sum(exp(trainscored), 2);
    
    %diff = trainscored(:,1) - trainscored(:,2);
         
    dh = contour( xvals, yvals, reshape(probs1, N_1d, N_1d ), 10, ...
        'LineWidth', .5); hold on;

    % Make plot prettier.
    set(gcf, 'color', 'white');
    set(gca, 'YGrid', 'off');
    %set(gca, 'Xtick', []);
    %set(gca, 'Ytick', []);
end

