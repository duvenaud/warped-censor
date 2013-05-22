%##########################################################################
% ExchangeSamp.m
% Do inference using exchange sampling, from Iain Murray's PhD thesis.
% THIS SAMPLER IS WRONG.
%##########################################################################

%##########################################################################
% Load or generate data
%##########################################################################
RequiredVars = {'Y'; 'X'; 'outD'; 'noiseVar'; 'covfunc'; 'likfunc'; ...
    'hyp'; 'Yo'; 'No'; 'XoTrue'; 'YcTrue'; 'XcTrue'; 'NcTrue'; 'N'; ...
    'latD'; 'outD'; 'pTruncEval'; 'pTruncSamp'};
clearVarsExcept;
close all;

existAll;
if (~allExist && existAllI == 1)
    fprintf('No data loaded... generating new data...\n');
    GenTruncData;
elseif(~allExist)
    fprintf('Partial data loaded. Figure out what you want to do.\n');
    return;
end

%##########################################################################
% Print the run's parameters
%##########################################################################
fprintf('Running LatentMC.m...\n');
fprintf('Censored points: %f%%\n', NcTrue / N * 100);
assert(No == size(Yo, 1));
assert(No == size(XoTrue, 1));
assert(size(YcTrue, 1) == NcTrue);
assert(size(XcTrue, 1) == NcTrue);

%##########################################################################
% Initialise the Markov Chain
%
% Expected variables
% Data           : Yo
% Initialisation : Xo
%##########################################################################

% Just as a test, initialise to the 'ground truth', from which we
% generated.
Xo = XoTrue;

% Alternatively, initialise to some random starting point:
% Xo = randn(size(Yo, 1), latD);

% Define proposal distribution for latent points
qXstd = 0.015;
qXEval = @(Xnew, Xold) (mvnpdf(Xnew, Xold, qXstd.^2 * eye(size(Xnew, 1))));
qXSamp = @(Xold) (Xold + randn(size(Xold)) * qXstd);

%##########################################################################
% Run the Markov Chain
%##########################################################################
% Variables for Markov Chain statistics
iteration = 1;
timeGap = 1;
nextTime = timeGap;
prevDrawIter = 1;

avgExtDataIters = 0;
acceptances = 0;
a_hist = [];

% Setup figures
figure(1);

tic;
while (1)
    XoDash = qXSamp(Xo);
    
    % Generate new extended data
    YExt = gpSamplePrior(XoDash, covfunc, hyp, outD);
    trunc = pTruncSamp(YExt);
    extIter = 1;
    while (sum(trunc) ~= 0)
        YExt = gpSamplePrior(XoDash, covfunc, hyp, outD);
        trunc = pTruncSamp(YExt);
        extIter = extIter + 1;
    end
    
    KXo = feval(covfunc, hyp.cov, Xo) + noiseVar * eye(size(Xo, 1));
    KXoDash = feval(covfunc, hyp.cov, XoDash) + noiseVar * eye(size(XoDash, 1));
    
    % Evaluate acceptance probability
%     qr = qXEval(XoDash, Xo) / qXEval(Xo, XoDash);
    lqr = log(1);
    lpXr = mvnlogpdf(XoDash, 0, 1) - mvnlogpdf(Xo, 0, 1);
    lpYr = zeros(size(Yo, 2), 1);
    lpYExtr = zeros(size(Yo, 2), 1);
    for d = 1:size(Yo, 2)
        lpYr(d) = mvnlogpdf(Yo(:, d), 0, KXoDash) - mvnlogpdf(Yo(:, d), 0, KXo);
        lpYExtr(d) = mvnlogpdf(YExt(:, d), 0, KXo) / mvnlogpdf(YExt(:, d), 0, KXoDash);
    end
    
    a = exp(lqr + lpXr + sum(lpYExtr) + sum(lpYr));
    
    if (rand(1) < a)
        acceptances = acceptances + 1;
    else
%         fprintf('n');
    end
    
    %######################################################################
    % Keep track of Markov Chain statistics
    %######################################################################
    avgExtDataIters = avgExtDataIters + extIter;
    a_hist = [a_hist; a];
    
    %######################################################################
    % Plot Markov chain statistics
    %######################################################################
    if ((toc - nextTime) > 0)
        iterDiff = iteration - prevDrawIter;
        
        % Draw some graphs
        set(0, 'CurrentFigure', 1);
        plot(min(a_hist, 1));
        drawnow;
        
        % Output some stats
        toc;
        fprintf('Average iterations to generate from P(Yo|Xo): %f\n', avgExtDataIters / iterDiff);
        fprintf('Acceptance ratio                            : %f\n', acceptances / iteration);
        fprintf('Iterations per second                       : %f\n', iterDiff / (toc - nextTime + timeGap));
        fprintf('\n');
        genExtDataIters = 0;
        
        % Setup stuff for next display time
        prevDrawIter = iteration;
        timeGap = timeGap * 1.1;
        nextTime = nextTime + timeGap;
    end
    
    iteration = iteration + 1;
end

%fprintf('qr    : %d\n', qr);
%fprintf('pXr   : %d\n', pXr);
%fprintf('pYr   : %d\n', pYr);
%fprintf('pyExtr: %d\n', pYExtr);