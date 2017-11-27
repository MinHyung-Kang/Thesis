clear;
close all;
N = 10000;
%mu = [-1;1];
mu = [-2;2];
mu2 = [-4;1];
mu3 = [-2;4];
sigma = ones(1,1,2);
w = [1/3;2/3];
w2 = [2/5;3/5];
w3 = [1/2;1/2];



gmm_model = gmdistribution(mu, sigma, w);
gmm_model2 = gmdistribution(mu2, sigma, w2);
gmm_model3 = gmdistribution(mu3, sigma, w3);


xVals = -10:0.1:10;
yVals = pdf(gmm_model,xVals');
yVals2 = pdf(gmm_model2, xVals');
yVals3 = pdf(gmm_model3, xVals');

figure;
subplot(1,2,1);
plot(xVals, yVals, 'LineWidth',3);
hold on;
plot(xVals, yVals2, ':','LineWidth',3);
axis off;
subplot(1,2,2);
plot(xVals, yVals, 'LineWidth',3);
hold on;
plot(xVals, yVals3, '--','LineWidth',3);
axis off;