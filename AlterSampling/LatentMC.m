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
predSamplesCull = 800;          % Number of samples to characterise the
                                % predictive distribution with.
timeGap = 30 * 60;
timeFactor = 1.1;

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
Xc = [];            % No need to init Nc, taken from size(Xc) later on.
Yc = [];

% Define proposal distribution for the censored points
% For now, choose the observed truncation rate as the parameter for the
% nbinobial distribution.
qNcEval = @(NcNew, NcOld) (nbinpdf(NcNew, No, No / (NcOld + No + 1)));
qNcSamp = @(NcOld) (nbinrnd(No, No / (NcOld + No + 1)));
% qNcEval = @(NcNew, NcOld) (nbinpdf(NcNew, No, 1 - NcTrue / N));
% qNcSample = @(NcOld) (nbinrnd(No, 1 - NcTrue / N));
% plot(qNcEval(1:300, 20));

% Define proposal distribution for latent points
qXostd = 0.002;
qXoEval = @(Xnew, Xold) (mvnpdf(Xnew, Xold, qXostd.^2 * eye(size(Xnew, 1))));
qXoSamp = @(Xold) (Xold + randn(size(Xold)) * qXostd);

qXcstd = 0.05;
qXcEval = @(Xnew, Xold) (mvnpdf(Xnew, Xold, qXcstd.^2 * eye(size(Xnew, 1))));
qXcSamp = @(Xold) (Xold + randn(size(Xold)) * qXcstd);

% Alternative proposal distribution for the censored latent points. Sample
% a random subset of the censored points, then perturb them.
% qXcEval = @SubsetPerturbEval;
% qXcSamp = @SubsetPerturbSamp;

%##########################################################################
% Run the Markov Chain
%
% Markov chain state consists of: Xo, Xc, Yc (and Nc, implied by the size
% of Xc.
%##########################################################################
% Stuff for the loop
nextTime = timeGap;
iteration = 1;

% Statistics of the Markov Chain
prevNumAccepted = 0;
previter = 0;
numAccepted = 0;
a_hist = [];
Nc_hist = [];
Xacc = [];
Yacc = [];
YaccFactor = 1;
XaccFactor = 1;

% Setup required figures
figure(2);
figure(3);
figure(4);
figure(5);
figure(6);
figure(7);
figure(8);

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
    a(1) = ( qNcEval(Nc, NcProp) / qNcEval(NcProp, Nc) ) * ...
        exp(gammaln(Nc + 1) + gammaln(NcProp + No) - gammaln(NcProp + 1) - gammaln(Nc + No) + truncLogProb);
    
    if (a(1) >= rand(1))
        accepted(1) = 1;
        if (NcProp > Nc)
            Xc = [Xc; XcAdd];
            Yc = [Yc; Yadd];
        elseif(NcProp < Nc)
            Xc(XcRemIndices, :) = [];
            Yc(XcRemIndices, :) = [];
        end
        
        Nc = NcProp;
    else
        accepted(1) = 0;
    end
    
    %######################################################################
    % Part 2: Resample latent variables Xc.
    %######################################################################
    if (size(Xc, 1) ~= 0)
        XcProp = qXcSamp(Xc);
        YcProp = gpSamplePosterior([Yo; Yc], [Xo; Xc], XcProp, covfunc, hyp);
        
        laPt2 = sum(mvnlogpdf(XcProp', 0, eye(latD))) - sum(mvnlogpdf(Xc', 0, eye(latD))) + ...
            sum(log(pTruncEval(YcProp))) - sum(log(pTruncEval(Yc))); % + ...
        %             log(qXcEval(Xc, XcProp)) - log(qXcEval(XcProp, Xc));
        a(2) = exp(laPt2);
        
        if (a(2) >= rand(1))
            accepted(2) = 1;
            Xc = XcProp;
            Yc = YcProp;
        else
            accepted(2) = 0;
        end
    else
        accepted(2) = 0;
    end
    
    %######################################################################
    % Part 3: Modifying the rejected values Yc.
    %######################################################################
    if (size(Xc, 1) ~= 0)
        YcProp = gpSamplePosterior(Yo, Xo, Xc, covfunc, hyp);
        
        a(3) = exp( sum(log(pTruncEval(YcProp))) - sum(log(pTruncEval(Yc))) );
        
        % Plot proposal
        % plot(Xc, YcProp(:, 1:2), 'x');
        
        if (a(3) >= rand(1))
            accepted(3) = 1;
            Yc = YcProp;
        else
            accepted(3) = 0;
        end
    else
        accepted(3) = 0;
    end
    
    %######################################################################
    % Part 4: Modifying the rejected latent points Xo.
    %######################################################################
    XoProp = qXoSamp(Xo);
    K_XoXc = feval(covfunc, hyp.cov, [Xo; Xc]) + eye(Nc + No) * noiseVar;
    K_XoPropXc = feval(covfunc, hyp.cov, [XoProp; Xc]) + eye(Nc + No) * noiseVar;
    
    a(4) = exp(sum(mvnlogpdf([Yo; Yc], 0, K_XoPropXc)) - sum(mvnlogpdf([Yo; Yc], 0, K_XoXc)));
    
    if (a(4) >= rand(1))
        accepted(4) = 1;
        Xo = XoProp;
    else
        accepted(4) = 0;
    end
    
    assert(sum(isnan(a)) == 0);
    
    %######################################################################
    % Accumulate results for predictive densities etc...
    %######################################################################
    if (mod(iteration, 200) == 1)
        Xall = [Xo; Xc];
%         Yall = [Yo; Yc];
%         Yacc = [Yacc; Yall(randperm(size(Yall, 1), ceil(size(Yall, 1) / YaccFactor)), :)];
%         
%         if (size(Yacc, 1) > 20000)
%             YaccFactor = YaccFactor * size(Yacc, 1) / 10000;
%             Yacc = Yacc(randperm(size(Yacc, 1), 10000), :);
%         end
        Ns = 800;
        Xs = randn(Ns, latD);
        Ys = gpSamplePosterior([Yo; Yc], [Xo; Xc], Xs, covfunc, hyp);
        
        Yacc = [Yacc; Ys(randperm(Ns, ceil(Ns / YaccFactor)), :)];
        if (size(Yacc, 1) > predSamplesCull)
            YaccFactor = YaccFactor * 2;
            Yacc = Yacc(randperm(size(Yacc, 1), predSamplesCull / 2), :);
        end
        
        Xacc = [Xacc; Xall(randperm(size(Xall, 1), ceil(size(Xall, 1) / XaccFactor)), :)];
        if (size(Xacc, 1) > predSamplesCull)
            XaccFactor = XaccFactor * 2;
            Xacc = Xacc(randperm(size(Xacc, 1), predSamplesCull / 2), :);
        end
    end
    
    %######################################################################
    % Gather statistics of Markov Chain
    %######################################################################
    numAccepted = numAccepted + accepted;
    a_hist = [a_hist; a];
    Nc_hist = [Nc_hist; Nc];
    
    %######################################################################
    % END OF MARKOV CHAIN - Loop and drawing bits and pieces
    %######################################################################
    if (toc - nextTime) > 0
        % Output statistics
        toc;
        fprintf('Iteration       : %i\n', iteration);
        fprintf('Iterations/s    : %f\n', (iteration - previter) / timeGap);
        fprintf('Accepted        : %i %i %i %i\n', numAccepted(1), numAccepted(2), numAccepted(3), numAccepted(4));
        fprintf('Acception rate  : %f %f %f %f%%\n\n', (numAccepted / iteration) * 100);
        fprintf('Recent accepted : %i %i %i %i\n', numAccepted - prevNumAccepted);
        fprintf('\n');
        
        %##################################################################
        % Plot Markov chain statistics
        %##################################################################
        set(0, 'CurrentFigure', 2);
        subplot(2, 1, 1);
        plot(log10(min(a_hist, 1)));
        title('Acceptance probability (Pt1)');
        ylim([-4, 0]);
        
        subplot(2, 2, 3);
        plot(Nc_hist);
        hold on; plot([1, length(Nc_hist)], [NcTrue, NcTrue]); hold off;
        NcPlotYlim = ylim;
        
        title('Number of censored points');
        subplot(2, 2, 4);
        [counts bins] = hist(Nc_hist);
        barh(bins, counts);
        ylim(NcPlotYlim);
        
        %##################################################################
        % Plot the state of the GPLVM
        %##################################################################
        set(0, 'CurrentFigure', 4);
        % Plot the observed and censored data in the output space
        if (size(Yo, 2) == 3)
            if (size(Yc, 2) ~= 0)
                plot3(Yo(:, 1), Yo(:, 2), Yo(:, 3), 'x', Yc(:, 1), Yc(:, 2), Yc(:, 3), 'o');
            else
                plot3(Yo(:, 1), Yo(:, 2), Yo(:, 3), 'x');
            end
        elseif (size(Yo, 2) == 2)
            if (size(Yc, 2) ~= 0)
                plot(Yo(:, 1), Yo(:, 2), 'x', Yc(:, 1), Yc(:, 2), 'o');
            else
                plot(Yo(:, 1), Yo(:, 2), 'x');
            end
        end
        title('Observed space');
        legend('Observed points', 'Current censored samples', 'Location', 'NorthWest');
        
        % Plot the mappings from the latent space
        if (size(Xo, 2) == 1)
            assert(size(Yo, 2) == 3);
            [Xsorted, Xpermutation] = sort(X);
            [XoSorted, XoPermutation] = sort(Xo);
            [XcSorted, XcPermutation] = sort(Xc);
            
            for od=1:outD
                set(0, 'CurrentFigure', 4 + od);
                subplot(3, 1, [1;2]);
                if (size(Yc, 2) ~= 0)
                    plot(Xsorted, Y(Xpermutation, od), XoSorted, Yo(XoPermutation, od), 'o', XcSorted, Yc(XcPermutation, od), 'x');
                else
                    plot(Xsorted, Y(Xpermutation, od), XoSorted, Yo(XoPermutation, od), 'o');
                end
                legend('Ground truth mapping', 'Observed', 'Current censored samples', 'Location', 'NorthEast');
                title(['Latent mapping ', num2str(od)]);
                xlabel('X');
                ylabel(['Y', num2str(od)]);
                
                subplot(3, 1, 3);
                bounds = xlim;
                r = bounds(1):0.01:bounds(2);
                [~, bincentres] = hist([Xo; Xc], 30);
                hist([Xo; Xc], 20);
                hold on;
                p = plot(r, mvnpdf(r', 0, 1) * size([Xo; Xc], 1) * (bincentres(4) - bincentres(3)));
                set(p, 'Color', 'red');
                hold off;
%                 yl = ylim;
%                 [nxo blo] = hist([Xo; Xc]);
%                 nxo = nxo / max(nxo) * yl(2) / 3;
%                 bar(blo, nxo);
            end
            
        elseif(size(Xo, 2) == 2)
            [xq, yq] = meshgrid(-4:.2:4, -4:.2:4);
            
            for od=1:outD
                Xs = [Xo; Xc];
                Ys = [Yo; Yc];
                
                YGrid = griddata(X(:, 1), X(:, 2), Y(:, od), xq, yq);
                YsGrid = griddata(Xs(:, 1), Xs(:, 2), Ys(:, od), xq, yq);
                
                set(0, 'CurrentFigure', 4 + od);
                mesh(xq, yq, YGrid, zeros(size(YGrid)));
                hold on;
                mesh(xq, yq, YsGrid, zeros(size(YsGrid)) + 1);
                alpha(0.5);
                plot3(XoTrue(:, 1), XoTrue(:, 2), Yo(:, od), 'x', ...
                    Xc(:, 1), Xc(:, 2), Yc(:, od), 'o');
                hold off;
                legend('Ground truth mapping', 'Currently sampled mapping', 'Ground truth points', 'Currently sampled points', 'Location', 'NorthWest');
            end
        end
        
        % Plot the distribution in the latent space
        set(0, 'CurrentFigure', 8);
        if (size(Xo, 2) == 1)
            hist(Xacc, 30);
            [~, bincentres] = hist(Xacc, 30);
            hold on;
            bounds = xlim;
            r = bounds(1):0.01:bounds(2);
            p = plot(r, mvnpdf(r', 0, 1) * size(Xacc, 1) * (bincentres(4) - bincentres(3)));
            set(p, 'Color', 'red');
            hold off;
        elseif (size(Xo, 2) == 2)
            plot(Xacc(:, 1), Xacc(:, 2), 'x');
        end
        
        % Plot the predictive distribution
        set(0, 'CurrentFigure', 3);
        if (outD == 3)
            plot3(Yacc(:, 1), Yacc(:, 2), Yacc(:, 3), 'x');
        end
        
        tilefigs([2 2], 0, 2, (2:3)');
        tilefigs([2 3], 0, 1, (4:8)');
        drawnow;
        
        % Setup next display loop
        timeGap = min(timeGap * timeFactor, 3600);
        nextTime = toc + timeGap;
        prevNumAccepted = numAccepted;
        previter = iteration;
    end
    
    iteration = iteration + 1;
end