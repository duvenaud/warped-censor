% CalcLiklihood.m
tic;
% gplvm;

% Importance samples for Pc
T = 10000000;

% Censor some values
observed = (Y(:, 1) > 0) & (Y(:, 2) > 0);
No = sum(observed);
XoTrue = X(observed, :);
Yo = Y(observed, :);

NcMax = 60;
Pc = zeros(NcMax, 1);
WPc = zeros(NcMax, 1);
for Nc = 44:NcMax
    KXo = covSEiso(hyp.cov, XoTrue) + noiseLevel * eye(No);
    
    for t=1:T
        % Sample Xc
        XcSamp = randn(Nc, latD);
        
        % Sample Yc (GP posterior)
        KXoXc = covSEiso(hyp.cov, XoTrue, XcSamp);
        KXcXc = covSEiso(hyp.cov, XcSamp);
        alpha = KXoXc' / KXo;
        cov = KXcXc - KXoXc'/KXo*KXoXc;
        R = chol(cov + noiseLevel * eye(Nc));
        
        Yc = zeros(Nc, d);
        for d=1:outD
            Yc(:, d) = alpha * Yo(:, d) + R'*randn(Nc, 1);
        end
        
        % Censoring function
        Pc(Nc) = Pc(Nc) + min((Yc(:, 1) > 0) & (Yc(:, 2) > 0));
    end
    Pc(Nc) = Pc(Nc) / T;
    WPc(Nc) = Pc(Nc) * nchoosek(Nc + N, Nc);
    disp(Nc);
    drawnow;
end

figure(1);
plot(1:NcMax, Pc);
figure(2);
plot(1:NcMax, WPc);

tilefigs;
toc;