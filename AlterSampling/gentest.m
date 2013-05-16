reqTruncSamples = 100000;

% Unnormalised density
l = 0.3;
r = 0.7;
if (~exist('h', 'var'))
    h = 1;
end
var = 0.5;
sampDensityU = @(x) ((x <= l | x > r) .* exp(-x.^2 / (2*var)) + (x > l & x <= r) * h) / sqrt(2*pi*var);

% Truncation function p(c=1|y)
truncFunc = @(x) (x > l & x <= r);

% Proposal density rough check, works because the sample density is similar
% to a Gaussian
sigma = sqrt(0.6);
propDensityN = @(x) normpdf(x, 0, sigma);
checkx = -4:0.1:4;
c = 1.01 * max(sampDensityU(checkx) ./ propDensityN(checkx));
propDensityU = @(x) c*propDensityN(x);

% Plot
figure(1);
ezplot(sampDensityU, [-3, 3]);
hold on;
ezplot(propDensityU, [-3, 3]);
hold off;
ylim([0, 0.7]);
drawnow;

% Sample from densityU using rejection sampling
samplesTrunc = [];
truncIters = 0;
rejIters = 0;
while (length(samplesTrunc) < reqTruncSamples)
    % Generate sample from underlying distribution (Step 1):
    samp = randn(1) * sigma;
    while (unifrnd(0, propDensityU(samp)) > sampDensityU(samp))
        samp = randn(1) * sigma;
        rejIters = rejIters + 1;
    end
    
    % Censor with P(c=1|y) (Step 2):
    % If not censored, store (Step 3):
    if (unifrnd(0, 1) > truncFunc(samp))
        samplesTrunc = [samplesTrunc; samp];
    end
    truncIters = truncIters + 1;
end

fprintf('Iterations per truncated point: %f\n', truncIters / reqTruncSamples);
fprintf('Rejections per latent sample  : %f\n', rejIters / truncIters);

figure(2);
hist(samplesTrunc, -3:0.1:3);
xlim([-3, 3]);