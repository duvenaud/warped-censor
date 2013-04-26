function [x, nll, acceptance_rate] = ea_hmc(likefunc, x, startnll, options, varargin)     
% Exact Approximate Hamiltonian Monte Carlo
%
% Based off of the implementation in David Mackay's book.
%
%
% Inputs:
%       likefunc:  Returns nll, dnll, takes (x, varargin)
%                  Can return a noisy but unbiased estimate of nll.
%                  dnll must still be exact.
%       x:         The starting state
%       startnll:  The old estimate of nll.  Can be NaN to start.
%       options:   A struct containing HMC params.
%       varargin:  Extra arguments to be passed to likefunc.
%
% options.Tau is the number of leapfrog steps.
% options.epsilon is step length
%
% Outputs:
%       x_new:     the new sampled state
%       nll:       The approximate nll of x_new
%     
% 
% David Duvenaud (dkd23@cam.ac.uk)
% Mark van der Wilk (mv310@cam.ac.uk)
%
% April 2013
% ===================================

num_acceptances = 0; %acceptance rate
L = 1;     %options.num_iters;

[nll, g] = likefunc( x, varargin{:});

% Use existing startnll if it exists.
if numel(startnll) > 0 && ~isnan(startnll)
    nll = startnll;
end

for l = 1:L
    momentum = randn( size( x ) );   % Randomly sample momentum term.
    hamiltonian = momentum' * momentum / 2 + nll;
    
    xnew = x; gnew = g;
 
    % Randomize number of steps and step size.
    cur_tau = randi(options.tau);
    cur_eps = rand * options.epsilon;
    %cur_tau = options.Tau;
    %cur_eps = options.epsilon;
    
    % Take a path from the starting state according to the momentum.
    for tau = 1:cur_tau
        momentum = momentum - cur_eps * gnew / 2;
        xnew = xnew + cur_eps * momentum;
        [ignore, gnew] = likefunc( xnew, varargin{:}); 
               
        momentum = momentum - cur_eps * gnew / 2;
    end
    
    % Evaluate likelihood at new location.
    [Enew, ignore] = likefunc( xnew, varargin{:});    
    Hnew = momentum' * momentum / 2 + Enew;
    
    % Metropolis-Hasting step.
    change_in_energy = Hnew - hamiltonian;
    
    % Accept with probability min(1, exp(-change_in_energy))
    % Ratio of proposals should be one, so we don't have to worry about it.
    if change_in_energy < 0
        accept = 1;
        fprintf('a');
    else
        if rand() < exp(-change_in_energy)
            accept = 1;
            fprintf('A');
        else
            accept = 0;
            fprintf('r');
        end
    end
    
    % Move to the new state.
    if accept
        g = gnew;
        x = xnew;
        nll = Enew;
        num_acceptances = num_acceptances+1;
    end
end

acceptance_rate = num_acceptances/L;
