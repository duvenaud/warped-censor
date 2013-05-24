SAMPLES = 300;

Y = mvnrnd([0 0], [2 0 ; 0 1], SAMPLES);
censor_is = rand(SAMPLES,1) < normpdf(Y(:,1), 0, 0.3);
Y(censor_is,:) = [];

scatter(Y(:,1), Y(:,2)); hold on;
clearvars -except Y;

save('gaussian_trunc');
clear;