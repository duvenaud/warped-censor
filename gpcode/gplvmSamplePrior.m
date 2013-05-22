% GP-LVM prior

function [Y, X] = gplvmSamplePrior (N, outD, latD, noiseLevel, covfunc, hyp)

if (~exist('N', 'var'))
    N = 1000;
end
if (~exist('outD', 'var'))
    outD = 3;
end
if (~exist('latD', 'var'))
    latD = 2;
end
if (~exist('noiseLevel', 'var'))
    noiseLevel = 0.0009;
end
if (~exist('hyp', 'var'))
    hyp.cov = [0 0];
end
if (~exist('covfunc', 'var'))
    covfunc = @covSEiso;
end

X = randn(N, latD);
K = feval(covfunc, hyp.cov, X);

Y = zeros(N, outD);
R = chol(K + noiseLevel * eye(N));
for d=1:outD
    Y(:, d) = R' * randn(N, 1);
%     if (d == 1)
%         yf1 = Y(:, d);
%     elseif(d == 2)
%         yf2 = Y(:, d);
%     elseif(d == 3)
%         yf3 = Y(:, d);
%     end
end

% close 1;
% figure(1);
% plot3(yf1, yf2, yf3, 'x'); axis tight;
% figure(2);
% plot(yf1, yf2, 'x');
% figure(3);
% hist(yf1, -3:0.1:3);
% axis tight;
% while true; camorbit(0.9,-0.1); drawnow; end

end