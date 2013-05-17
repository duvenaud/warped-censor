%##########################################################################
% LatentMC.m
% Do sampling à la Ryan Adams' GPDS paper in the truncation model.
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
qXstd = 0.01;
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
numAcceptedPt1 = 0;
numAcceptedPt2 = 0;
prevNumAccepted = 0;
previter = 0;
aPt1_hist = [];
aPt2_hist = [];
Nc_hist = [];

% Setup required figures
figure(2);
figure(3);
figure(4);
figure(5);
figure(6);
figure(7);

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
%     XoProp = qXSamp(Xo);
    XcProp = qXSamp(Xc);
    
    % Draw new GP values at the perturbed points from P(Yhat|X, Y, Xpert)
    YcHat = gpSamplePosterior([Yo; Yc], [XoProp; Xc], XcProp, covfunc, hyp);
    
    % Perturb GP mapping using Ypert = alpha * Yhat + sqrt(1 - alpha^2) *
    % Yp.
    % Yp ~ P(Y|Xpert)
    YcPert = zeros(size(Xc, 1), size(YcHat, 2));
    [YcPert(:, 1) R] = gpSamplePrior(Xc, covfunc, hyp);
    YcPert(:, [2, 3]) = R'* randn(size(YcPert, 1), size(YcPert, 2) - 1);
    YcProp = MCPt2ProposalRelaxation * YcHat + sqrt(1-MCPt2ProposalRelaxation^2) * YcPert;
    
    % Accept with correct probability.
    aPt2 = ( qXEval(XoProp, Xo) * qXEval(XcProp, Xc) ) / ...
        ( qXEval(Xo, XoProp) * qXEval(Xc, XcProp) ) * ...
        mvnpdf(XcProp) * prod(pTruncEval(YcProp)) / ...
        mvnpdf(Xc)     * prod(pTruncEval(Yc));
    
    if (aPt2 >= rand(1))
        acceptedPt2 = 1;
        
        Yc = YcProp;
        Xo = XoProp;
        Xc = XcProp;
    else
        acceptedPt2 = 0;
    end
    
    %######################################################################
    % Gather statistics of Markov Chain
    %######################################################################
    numAcceptedPt1 = numAcceptedPt1 + acceptedPt1;
    numAcceptedPt2 = numAcceptedPt2 + acceptedPt2;
    aPt1_hist = [aPt1_hist; aPt1];
    Nc_hist = [Nc_hist; Nc];
    aPt2_hist = [aPt2_hist; aPt2];
    
    %######################################################################
    % END OF MARKOV CHAIN - Loop and drawing bits and pieces
    %######################################################################
    if (toc - nextTime) > 0
        % Output statistics
        toc;
        fprintf('Iteration       : %i\n', iteration);
        fprintf('Iterations/s    : %f\n', (iteration - previter) / timeGap);
        fprintf('Accepted        : %i\n', numAcceptedPt1);
        fprintf('Acception rate  : %f%%\n\n', (numAcceptedPt1 / iteration) * 100);
        fprintf('Recent accepted : %i\n', numAcceptedPt1 - prevNumAccepted);
        fprintf('\n');
        
        %##################################################################
        % Plot Markov chain statistics
        %##################################################################
        set(0, 'CurrentFigure', 2);
        subplot(2, 1, 1);
        plot(log10(min(aPt1_hist, 1)));
        title('Acceptance probability (Pt1)');
        ylim([-4, 0]);
        
        subplot(2, 2, 3);
        plot(Nc_hist);
        title('Number of censored points');
        
        subplot(2, 2, 4);
        [counts bins] = hist(Nc_hist);
        barh(bins, counts);
        
        set(0, 'CurrentFigure', 3);
        plot(log10(min(aPt2_hist, 1)));
        title('Acceptance probability (Pt2)');
        ylim([-4, 0]);
        
        %##################################################################
        % Plot the state of the GPLVM
        %##################################################################
        set(0, 'CurrentFigure', 4);
        % Plot the observed and censored data in the output space
        if (size(Yo, 2) == 3)
            plot3(Yo(:, 1), Yo(:, 2), Yo(:, 3), 'x', Yc(:, 1), Yc(:, 2), Yc(:, 3), 'o');
        elseif (size(Yo, 2) == 2)
            plot(Yo(:, 1), Yo(:, 2), 'x', Yc(:, 1), Yc(:, 2), 'o');
        end
        
        % Plot the mappings from the latent space
        if (size(Xo, 2) == 1)
            assert(size(Yo, 2) == 3);
            [Xsorted, Xpermutation] = sort(X);
            [XoSorted, XoPermutation] = sort(Xo);
            [XcSorted, XcPermutation] = sort(Xc);
            
            set(0, 'CurrentFigure', 5);
            plot(Xsorted, Y(Xpermutation, 1), XoSorted, Yo(XoPermutation, 1), XcSorted, Yc(XcPermutation, 1));
            set(0, 'CurrentFigure', 6);
            plot(Xsorted, Y(Xpermutation, 2), XoSorted, Yo(XoPermutation, 2), XcSorted, Yc(XcPermutation, 2));
            set(0, 'CurrentFigure', 7);
            plot(Xsorted, Y(Xpermutation, 3), XoSorted, Yo(XoPermutation, 3), XcSorted, Yc(XcPermutation, 3));
        end
        
        tilefigs([2 2], 0, 2, (2:3)');
        tilefigs([2 3], 0, 1, (4:7)');
        drawnow;
        
        % Setup next display loop
        timeGap = min(timeGap * 1.1, 3600);
        nextTime = toc + timeGap;
        prevNumAccepted = numAcceptedPt1;
        previter = iteration;
    end
    
    iteration = iteration + 1;
end