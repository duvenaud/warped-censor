MEAN = [0 0];
COVARIANCE = [1 .66 ; .66 1];
SAMPLES = 500;
MIN_CENSOR_X = -0.25;
MIN_CENSOR_Y = -0.5;

Y = mvnrnd(MEAN, COVARIANCE, SAMPLES);

Y(Y(:,1) > MIN_CENSOR_X & Y(:,2) > MIN_CENSOR_Y, :) = [];

scatter(Y(:,1), Y(:,2));

clearvars -except Y;
save('gaussian');
clear;