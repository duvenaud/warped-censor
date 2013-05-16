% SampleSemiPred.m (prev. calcLiklihood)
% clear;

outD = 3;
latD = 1;
noiseLevel = 0.0009;
hyp.cov = [0 0];

% [Y X] = gplvmSamplePrior(1000, outD, latD, noiseLevel);
yf1 = Y(:, 1);yf2 = Y(:, 2);yf3 = Y(:, 3);
% load nicedata;
% load filamentdata;

%% Censor and display
cregion = (Y(:, 1) < 0) & (Y(:, 2) < 0);
censored = cregion * 0.99 > rand(size(Y, 1), 1);
% observed = (yf1 > 0) & (yf2 > 0);
observed = ~censored;

fprintf('Percent censored: %f\n', sum(censored) / (sum(observed) + sum(censored)) * 100);

yo1 = yf1(observed);yo2 = yf2(observed);yo3 = yf3(observed);
yc1 = yf1(censored);yc2 = yf2(censored);yc3 = yf3(censored);

figure(2); hold off;
plot3(yo1, yo2, yo3', 'x');
xlabel('yo1');ylabel('yo2');zlabel('yo3');
title('Observed points');

figure(3); hold off;
plot3(yo1, yo2, yo3, 'o', yc1, yc2, yc3, 'x');
axis tight;
a = axis;
axis(a);
xlabel('yo1');ylabel('yo2');zlabel('yo3');
title('Observed and censored points - GROUND TRUTH');

Nc = 1000;
NumObserved = sum(observed);
XcSamp = randn(Nc, size(X, 2));
XoTrue = X(observed, :);
% XoSamp = randn(sum(observed), size(x, 2));

%% Pc calculation
% Get posteriors distribution of the GP transformations | Yo, XoTrue, XcSamp
KXo = covSEiso(hyp.cov, XoTrue) + noiseLevel * eye(NumObserved);
KXoXc = covSEiso(hyp.cov, XoTrue, XcSamp);
KXcXc = covSEiso(hyp.cov, XcSamp, XcSamp);
mean1 = KXoXc' / KXo * yo1;
mean2 = KXoXc' / KXo * yo2;
mean3 = KXoXc' / KXo * yo3;
cov = KXcXc - KXoXc'/KXo*KXoXc;

R = chol(cov + noiseLevel * eye(size(cov)));
yc1s = mean1 + R' * randn(size(cov, 1), 1);
yc2s = mean2 + R' * randn(size(cov, 1), 1);
yc3s = mean3 + R' * randn(size(cov, 1), 1);

figure(4);
plot3(yo1, yo2, yo3, 'o', yc1s, yc2s, yc3s, 'x');
% plot3([0], [0], [0], 'x', yc1s, yc2s, yc3s, 'x');
axis(a);
title('Sample from P(Yl | Yo, XoTrue, XcSamp)');

if (size(X, 2) == 1)
    [s1, I1] = sort(XcSamp);
    [s2, I2] = sort(X);
    figure(5);
    plot(s1, yc1s(I1), s2, yf1(I2), XoTrue, yo1, 'x');
    figure(6);
    plot(s1, yc2s(I1), s2, yf2(I2), XoTrue, yo2, 'x');
    figure(7);
    plot(s1, yc3s(I1), s2, yf3(I2), XoTrue, yo3, 'x');
%     figure(8);
%     plot(yc3s(I1) - yf3(I2));
elseif (size(X, 2) == 2)
%     plot3(XcSamp(:, 1), XcSamp(:, 2), yc1s, 'x', x(:, 1), x(:, 2), yf1, 'x');
%     tri = delaunay(x(:, 1), x(:, 2));
%     trisurf(tri, x(:, 1), x(:, 2), yf1);

    [xq, yq] = meshgrid(-4:.2:4, -4:.2:4);
    
    figure(5);
    yf1r = griddata(X(:, 1), X(:, 2), yf1, xq, yq);
    yc1sr = griddata(XcSamp(:, 1), XcSamp(:, 2), yc1s, xq, yq);
    
%     plot3(x(:, 1), x(:, 2), yf1, 'x', XcSamp(:, 1), XcSamp(:, 2), yc1s, 'x', XoTrue(:, 1), XoTrue(:, 2), yo1, 'x');
    mesh(xq, yq, yf1r, zeros(size(yf1r)));
    hold on;
    mesh(xq, yq, yc1sr, zeros(size(yf1r)) + 1);
    alpha(0.5);
    plot3(XoTrue(:, 1), XoTrue(:, 2), yo1, 'x');
    hold off;
    
    figure(6);
    yf2r = griddata(X(:, 1), X(:, 2), yf2, xq, yq);
    yc2sr = griddata(XcSamp(:, 1), XcSamp(:, 2), yc2s, xq, yq);
    mesh(xq, yq, yf2r, zeros(size(yf2r)));
    hold on;
    mesh(xq, yq, yc2sr, zeros(size(yf2r)) + 1);
    plot3(XoTrue(:, 1), XoTrue(:, 2), yo2, 'x');
    hold off;
    alpha(0.5);
    
    figure(7);
    yf3r = griddata(X(:, 1), X(:, 2), yf3, xq, yq);
    yc3sr = griddata(XcSamp(:, 1), XcSamp(:, 2), yc3s, xq, yq);
    mesh(xq, yq, yf3r, zeros(size(yf3r)));
    hold on;
    mesh(xq, yq, yc3sr, zeros(size(yf3r)) + 1);
    plot3(XoTrue(:, 1), XoTrue(:, 2), yo3, 'x');
    hold off;
    alpha(0.5);
    
    figure(8);
    mesh(xq, yq, yf3r - yc3sr);
    hold on;
    hold off;
end