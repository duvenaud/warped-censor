%##########################################################################
% LatentMC.m
% Do sampling à la Ryan Adams' GPDS paper in the truncation model.
%##########################################################################

%##########################################################################
% Load or generate data
%##########################################################################
RequiredVars = {'Y'; 'X'; 'outD'; 'noiseLevel'; 'covfunc'; 'hyp'; 'Yo'; ...
                'No'; 'XoTrue'; 'YcTrue'; 'XcTrue'; 'NcTrue'; 'N'; ...
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
% Set parameters
%##########################################################################
MCPt2ProposalRelaxation = 0.95;

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
% Data           : Yo No
% Initialisation : Xo Xc Yc
%##########################################################################

% Just as a test, initialise to the 'ground truth', from which we
% generated.
Xo = XoTrue;
Xc = XcTrue;            % No need to init Nc, taken from size(Xc) later on.
Yc = YcTrue;

% Define proposal distribution for the censored points
% For now, choose the observed truncation rate as the parameter for the
% nbinobial distribution.
qNcEval = @(NcNew, NcOld) (nbinpdf(NcNew, No, No / (NcOld + No)));
qNcSamp = @(NcOld) (nbinrnd(No, No / (NcOld + No)));
% qNcEval = @(NcNew, NcOld) (nbinpdf(NcNew, No, 1 - NcTrue / N));
% qNcSample = @(NcOld) (nbinrnd(No, 1 - NcTrue / N));
% plot(qNcEval(1:300, 20));

% Define proposal distribution for latent points
qXstd = 0.1;
qXEval = @(Xnew, Xold) (mvnpdf(Xnew, Xold, qXstd.^2 * eye(size(Xnew, 1))));
qXSamp = @(Xold) (Xold + randn(size(Xold)) * qXstd);

%##########################################################################
% Run the Markov Chain
%
% Markov chain state consists of: Xo, Xc, Yc (and Nc, implied by the size
% of Xc.
%##########################################################################
% Stuff for the loop
timeGap = 1;
nextTime = timeGap;
iteration = 1;

% Statistics of the Markov Chain
numAccepted = 0;
prevNumAccepted = 0;
previter = 0;
aPt1_hist = [];
aPt2_hist = [];
Nc_hist = [];

% Setup required figures
figure(2);
figure(3);
figure(4);

tic;
while (1)
    Nc = size(Xc, 1);
    
    %######################################################################
    % PART 1: Resample the number of censored points.
    %######################################################################
    NcProp = qNcSamp(Nc);
    
    Yadd = [];
    if (NcProp > Nc)
        % Must add new truncated observations
        XcAdd = randn(NcProp - Nc, latD);
        % Sample the GP at these points (i.e. sample the observations Yl
        % before censoring)
        Yadd = gpSamplePosterior([Yo; Yc], [Xo; Xc], XcAdd, covfunc, hyp);
        % Calculate the probability of truncating for these observations.
        truncLogProb = sum(log(pTruncEval(Yadd)));
    elseif (NcProp < Nc)
        % Must remove...
        XcRemIndices = randperm(size(Xc, 1));
        XcRemIndices = XcRemIndices(1:(Nc - NcProp));
        truncLogProb = -sum(log(pTruncEval(Yc(XcRemIndices, :))));
    elseif (NcProp == Nc)
        truncLogProb = 0;
    else
        error('');
    end
    
    % a = truncProb * ( qNcEval(Nc, NcProp) / qNcEval(NcProp, Nc) ) * ...
    %     ( (factorial(Nc) * factorial(NcProp + No - 1)) / ...
    %       (factorial(NcProp) * factorial(Nc + No -1)) ...
    %     );
    
    % Alternative way of calculating the acceptance probability using
    % gammaln().
    aPt1 = ( qNcEval(Nc, NcProp) / qNcEval(NcProp, Nc) ) * ...
        exp(gammaln(Nc + 1) + gammaln(NcProp + No) - gammaln(NcProp + 1) - gammaln(Nc + No) + truncLogProb);
    
    if (aPt1 >= rand(1))
        acceptedPt1 = 1;
        if (NcProp > Nc)
            Xc = [Xc; XcAdd];
            Yc = [Yc; Yadd];
        elseif(NcProp < Nc)
            Xc(XcRemIndices, :) = [];
            Yc(XcRemIndices, :) = [];
        end
        
        Nc = NcProp;
    else
        acceptedPt1 = 0;
    end
    
    %######################################################################
    % Part 2: Resample latent variables and GP mappings.
    %######################################################################
    
    % Perturb Xpert = Xo Xc
    XoProp = qXSamp(Xo);
    XcProp = qXSamp(Xc);
    
    % Draw new GP values at the perturbed points from P(Yhat|X, Y, Xpert)
    Yhat = gpSamplePosterior([Yo; Yc], [Xo; Xc], [XoProp; XcProp], covfunc, hyp);
    
    % Perturb GP mapping using Ypert = alpha * Yhat + sqrt(1 - alpha^2) *
    % Yp.
    % Yp ~ P(Y|Xpert)
    Ypert = zeros(size(Xo, 1) + size(Xc, 1), size(Yhat, 2));
    [Ypert(:, 1) R] = gpSamplePrior([Xo; Xc], covfunc, hyp);
    Ypert(:, [2, 3]) = R'* randn(size(Ypert, 1), size(Ypert, 2) - 1);
    Yprop = MCPt2ProposalRelaxation * Yhat + sqrt(1-MCPt2ProposalRelaxation^2) * Ypert;
    YoProp = Yprop(1:No, :);
    YcProp = Yprop(No+1:end, :);
    
    % Accept with correct probability.
    aPt2 = ( qXEval(XoProp, Xo) * qXEval(XcProp, Xc) ) / ...
           ( qXEval(Xo, XoProp) * qXEval(Xc, XcProp) ) * ...
           mvnpdf([XoProp; XcProp]) * prod(1 - pTruncEval(YoProp)) * prod(pTruncEval(YcProp)) / ...
           mvnpdf([Xo; Xc]) * prod(1 - pTruncEval(Yo)) * prod(pTruncEval(Yc));
        
    %######################################################################
    % Gather statistics of Markov Chain
    %######################################################################
    numAccepted = numAccepted + acceptedPt1;
    aPt1_hist = [aPt1_hist; aPt1];
    Nc_hist = [Nc_hist; Nc];
    aPt2_hist = [aPt2_hist; aPt2];
    
    %######################################################################
    % END OF MARKOV CHAIN - Loop and drawing bits and pieces
    %######################################################################
    if (toc - nextTime) > 0
%     if (accepted)
        % Output statistics
        toc;
        fprintf('Iteration       : %i\n', iteration);
        fprintf('Iterations/s    : %f\n', (iteration - previter) / timeGap);
        fprintf('Accepted        : %i\n', numAccepted);
        fprintf('Acception rate  : %f%%\n\n', (numAccepted / iteration) * 100);
        fprintf('Recent accepted : %i\n', numAccepted - prevNumAccepted);
        fprintf('\n');
        
        % Draw code...
        set(0, 'CurrentFigure', 2);
        subplot(2, 1, 1);
        plot(log10(min(aPt1_hist, 1)));
        title('Acceptance probability');
        ylim([-4, 0]);
        
        subplot(2, 2, 3);
        plot(Nc_hist);
        title('Number of censored points');
        
        subplot(2, 2, 4);
        [counts bins] = hist(Nc_hist);
        barh(bins, counts);
        
        set(0, 'CurrentFigure', 3);
        plot3(Yo(:, 1), Yo(:, 2), Yo(:, 3), 'x', Yc(:, 1), Yc(:, 2), Yc(:, 3), 'o');
        
        set(0, 'CurrentFigure', 4);
        plot(min(aPt2_hist, 1));

        tilefigs;
        drawnow;
        
        % Setup next display loop
        timeGap = min(timeGap * 1.1, 3600);
        nextTime = toc + timeGap;
        prevNumAccepted = numAccepted;
        previter = iteration;
    end
    
    iteration = iteration + 1;
end