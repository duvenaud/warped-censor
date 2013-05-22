%##########################################################################
% CalcLiklihood.m
% Calculate the probability of all censored points being censored by
% sampling. In this case:
%   1. Sample Xc
%   2. Sample Yc from P(Yc|Yo Xo Xc)
%   3. Average p(c=1|Yc)
%##########################################################################
tic;

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
% Do calculation
%##########################################################################
NcMax = 60;
T = 100000;

Pc = zeros(NcMax, 1);
WPc = zeros(NcMax, 1);
KXo = covSEiso(hyp.cov, XoTrue) + noiseVar * eye(No);

for Nc = 1:NcMax
    for t=1:T
        % Sample Xc
        XcSamp = randn(Nc, latD);
        
        % Sample Yc (GP posterior)
        Yc = gpSamplePosterior(Yo, XoTrue, XcSamp, covfunc, hyp, KXo);
        
        % Censoring function
        Pc(Nc) = Pc(Nc) + prod(pTruncEval(Yc));
    end
    Pc(Nc) = Pc(Nc) / T;
    WPc(Nc) = Pc(Nc) * nchoosek(Nc + No - 1, Nc);
    disp(Nc);
    
%     figure(1);
%     plot(Pc);
%     figure(2);
%     plot(WPc);
%     tilefigs;
%     drawnow;
end


%% plot
figure(1);
plot(1:NcMax, Pc);
figure(2);
plot(1:NcMax, WPc);
figure(3);
plot(Pc(1:end-1) ./ Pc(2:end));

tilefigs;
toc;

sum(WPc);