%% Demo code to show mean/variance estimation of SVGD


%% Generate a random GMM
%rng(1);
clear;
close all;
N = 10000;
%mu = [-1;1];
%mu = [-4;3];
mu = [-3;3];
sigma = ones(1,1,2);
w = [1/3;2/3];
p_params.mu = mu;
p_params.sigma = sigma;
p_params.w = w;

true_x = 2/3;
true_xsq = 5;

gmm_model = gmdistribution(mu, sigma, w);

% Score function
dlog_p = @(X, seed)getGradGMM(gmm_model, p_params, X);
theta_nuts = random(gmm_model,N);
% Evaluation function
x_nuts = mean(theta_nuts,1);
xsq_nuts = mean(theta_nuts.^2,1);
%cov_nuts = cov(theta_nuts);

evalPerformance = @(points)evalPoints(points, x_nuts, xsq_nuts);
evalMMD = @(points)evalPointsMMD(points, theta_nuts);

%% Options

fudge_factor = 1e-6;

M = 100;
maxIter = 800;  % maximum iteration times
numTimeSteps = 10; % How many timesteps we want to shows
maxTrial = 10;% How many trials we want to average over
timeStepUnit = maxIter / numTimeSteps;
m = 20;
adverIter = 10;

baseModel = getModel('none',-1);                                   % Base Model
subsetModel = getModel('subset',-1, m);                            % Random Subset
subsetCFModel = getModel('subsetCF',-1, m);                        % Random Subset (CF)
inducedModel = getModel('inducedPoints',-1, m,0,-1,0,false);       % Induced Points
inducedAdverModel = getModel('inducedPoints',-1, m,3,adverIter,0.1,true);   % Induced Points - update y,h
inducedAdverModelSubset = getModel('inducedPoints',-1, m,3,adverIter,0.1,true,m);   % Induced Points - update y,h

algNames = {'SVGD','Random Subset', 'Random Subset + Control Functional', ...
    'Induced Points', 'Adversarial Induced Points (10 updates)','Adversarial Induced Points (batch,10 updates)'};

numModels = length(algNames);


%% Run iterations

mse_x = zeros(numModels,numTimeSteps);
mse_cov = zeros(numModels,numTimeSteps);
t_vals = zeros(numModels, numTimeSteps);
mmd_stat = zeros(numModels, numTimeSteps);
modelOpts = {baseModel, subsetModel, subsetCFModel, inducedModel, inducedAdverModel, inducedAdverModelSubset};

xVals = [-8:0.1:8];
yVals = pdf(gmm_model,xVals');
c = linspace(1,10,M);
if maxTrial == 1; figure('Position', [200, 100, 1000, 600]); end

for trialInd = 1:maxTrial
    % Common starting parameter
    %theta0 = normrnd(-10,1,M,1);
    %theta0 = normrnd(0,M,1);
    theta0 = normrnd(-10,1,M,1);
    currentSeed = trialInd;

    for modelInd = 1:numModels
        fprintf('Evaluating : Trial (%d/%d) Model : %s \n',...
            trialInd, maxTrial, algNames{modelInd});

        theta = theta0;
        modelOpt = modelOpts{modelInd};

        % Except monte carlo case

        modelOpt.currentSeed = currentSeed;

        gradInfo.historical_grad = 0;
        gradInfo.y_historical_grad = 0;
        gradInfo.h_historical_grad = 0;
        timePassed = 0;

        % Set initial points
        if modelOpt.kernel_opts.adver == 1 || modelOpt.kernel_opts.adver == 3
            Y = theta(randi([1,M],modelOpt.kernel_opts.m,1),:);
            modelOpt.kernel_opts.Y = Y;
        end

        if maxTrial == 1
            subplot(3,2,modelInd);
            plot(xVals,yVals,'b-','LineWidth',5);
            title(sprintf('%s\n',algNames{modelInd}));
            hold on;
            cfPlot=plot(1,1);
            cfPlot2 = plot(2,2);
            pause(1);
        end

        % Iterations
        for iter = 1:maxIter
            % Get update
            timeStart = tic;

            modelOpt.nextSeed = trialInd * maxIter + iter;
            [theta, gradInfo, modelOpt] = svgd_singleIteration(theta, dlog_p, gradInfo, modelOpt);


            timePassed = timePassed + toc(timeStart);

            if mod(iter, timeStepUnit) == 0 % Print and evaluate at current step
                [MSE_x, MSE_cov] = evalPerformance(theta);
%                 HInfo= evalMMD(theta);

                pause(0.1);
                if(maxTrial == 1)
                    delete(cfPlot);
                    delete(cfPlot2);
                    [f,xi] = ksdensity(theta);
                    [fPoints,~] = ksdensity(theta,theta);
                    cfPlot = plot(xi,f,'r:','LineWidth',4);
                    %cfPlot2 = scatter(theta,zeros(1,size(theta,1)),'ro','MarkerSize',4,'LineWidth',1);
                    cfPlot2 = scatter(theta,zeros(1,size(theta,1)),80,c,'filled');
                    title(sprintf('%s \n(Iteration %d)',algNames{modelInd},iter));
                    %cfPlot = plot(theta,ones(1,M)*0.1,'o');
                    %cfPlot = plot(gradInfo.ksdInfo.y,gradInfo.ksdInfo.w,'o');
                    drawnow;
                end

                iterUnit = iter / timeStepUnit;
                fprintf('Evaluating (%d/%d trials) : Model (%d/%d) - Iteration (%d/%d)\n', ...
                        trialInd, maxTrial, modelInd, numModels, iter, maxIter);
                t_vals(modelInd,iterUnit) = t_vals(modelInd,iterUnit) + timePassed / maxTrial;
                mse_x(modelInd,iterUnit) = mse_x(modelInd,iterUnit) + MSE_x / maxTrial;
                mse_cov(modelInd,iterUnit) = mse_cov(modelInd,iterUnit) + MSE_cov / maxTrial;
%                 mmd_stat(modelInd, iterUnit) = mmd_stat(modelInd, iterUnit) + HInfo.mmd.val/maxTrial;
            end
        end
    end
end



%% Plot results
results = {t_vals,mse_x,mse_cov, mmd_stat};
%% Plot result

figure;
colOpts = {'h-','o-','*-','.-','x-','s-','d-','^-','v-','p-','h-','>-','<-'};
% titleNames = {'Total Time', 'Mean', 'Covariance', 'Maximum Discrepancy'};
titleNames = {'Total Time', 'Mean', 'Covariance'};
valNames = {'log10 t','log10 MSE','log10 MSE'};
plotTime = timeStepUnit * (1:numTimeSteps);

for j = 1:3
    subplot(2,2,j);
    handles = zeros(1, numModels);
    result = results{j};

    for i = 1:numModels
        if j == 1
            handles(i) = semilogy(plotTime, result(i,:),colOpts{i});
        else
            handles(i) = plot(plotTime, result(i,:),colOpts{i});
        end
        hold on;
    end
    title(sprintf('%s',titleNames{j}));
    xlabel('Iterations');
    ylabel(valNames{j});
end

subplot(2,2,4);
axis off;
leg1 = legend(handles,algNames, 'Orientation','vertical');
set(leg1, 'Position',[0.75 0.3 0 0]);
%set(leg1, 'Position',[0.8 0.5 0.05 0.05]);






%% Auxiliary functions
function gradp = getGradGMM(gmm, p_params, X)
    [N, T] = size(X);
    pVal = pdf(gmm,X);
    [numComp,~] = size(p_params.mu);
    gradp = zeros(N, T);
    for k = 1:numComp % For each component
        mu_k = p_params.mu(k,:); covMat_i = p_params.sigma(:,:,k);
        w_k = p_params.w(k);
        front = mvnpdf(X, mu_k, covMat_i);   % n by 1
        back = (bsxfun(@minus, X, p_params.mu(k,:)) * (inv(covMat_i))'); % n by d
        gradp = gradp + (-1) * w_k * bsxfun(@times, front, back);
    end
    gradp = bsxfun(@rdivide, gradp, pVal);
end

%% Auxiliary functions
function [MSE_x, MSE_xsq] = evalPoints(points, x_nuts, xsq_nuts)
     x_hat = mean(points,1);
     xsq_hat = mean(points.^2,1);
     %cov_hat = cov(points);

     MSE = @(y,yhat)(mean((y(:) - yhat(:)).^2));

     MSE_x = log10(MSE(x_hat, x_nuts));
     MSE_xsq = log10(MSE(xsq_hat, xsq_nuts));
 end

 function [HInfo]= evalPointsMMD(points, points_nuts)
     points_total = [points;points_nuts];
     labels = [ones(1,size(points_nuts,1)) ones(1,size(points,1)) * -1];
     clear global Kxx;
     [~,HInfo]= mmd(points_total, labels);
 end

%{
function [RMSE_mu, RMSE_cov] = evalPoints(points, mu, cov)
    [mu_hat, mu_cov] = estimateGaussianMean(points);

    RMSE_mu = log10(RMSE(mu_hat, mu));
    RMSE_cov = log10(RMSE(mu_cov, cov));
end

function [mu_hat, mu_cov] = estimateGaussianMean(points)
    mu_hat = mean(points,1);
    points_n = size(points,1);
    covMat = (points - repmat(mu_hat,points_n,1));
    mu_cov = covMat' * covMat / points_n;
end
%}
