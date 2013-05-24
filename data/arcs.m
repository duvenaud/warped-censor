SAMPLES = 200;
CENSOR_WIDTH = 0.33;

%mean1 = [0 1];
%mean2 = [0 -1];
%covariance = [1 0 ; 0 .01];
%line1 = mvnrnd(mean1, covariance, SAMPLES/2);
%line2 = mvnrnd(mean2, covariance, SAMPLES/2);

width = 0.1;
line1 = rand(SAMPLES, 2) * [1 0 ; 0 0.1];
line2 = rand(SAMPLES, 2) * [1 0 ; 0 0.1];
line1 = [line1(:,1) - 0.5, line1(:,2) - width/2];
line2 = [line2(:,1) - 0.5, line2(:,2) - width/2];

arc1 = [line1(:,1), 0.5 + width - sqrt(0.25 - line1(:,1).^2) + line1(:,2)];
arc2 = [line2(:,1), -0.5 - width + sqrt(0.25 - line2(:,1).^2) + line2(:,2)];

data = [arc1 ; arc2];

data(data(:,1) > -CENSOR_WIDTH/2 & data(:,1) < CENSOR_WIDTH/2, :) = [];

scatter(data(:,1), data(:,2));

Y = data;
clearvars -except Y;
save('arcs');
clear;