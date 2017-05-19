%% Demo code to show mean/variance estimation of SVGD


%% Generate a random GMM
%rng(1);
clear;

%mu = [-1;1];
mu = [-2;2];
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
% Evaluation function
evalPerformance = @(points)evalPoints(points, true_x, true_xsq);

%% Options

fudge_factor = 1e-6;

M = 100;
maxIter = 1500;  % maximum iteration times
numTimeSteps = 10; % How many timesteps we want to shows
maxTrial = 1;% How many trials we want to average over
timeStepUnit = maxIter / numTimeSteps;
m = 20;
adverIter = 10;

baseModel = getModel('none',-1);                                   % Base Model
subsetModel = getModel('subset',-1, m);                            % Random Subset
subsetCFModel = getModel('subsetCF',-1, m);                        % Random Subset (CF)
inducedModel = getModel('inducedPoints',-1, m,0,-1,0,false);       % Induced Points
inducedAdverModel = getModel('inducedPoints',-1, m,3,adverIter,0.1,true);   % Induced Points - update y,h

algNames = {'SVGD','Random Subset', 'Random Subset + Control Functional', ...
    'Induced Points', 'Adversarial Induced Points (10 updates)'};

numModels = length(algNames);


%% Run iterations

mse_x = zeros(numModels,numTimeSteps);
mse_xsq = zeros(numModels,numTimeSteps);
t_vals = zeros(numModels, numTimeSteps);
modelOpts = {baseModel, subsetModel, subsetCFModel, inducedModel, inducedAdverModel};

% xVals = [-4:0.1:4];
% yVals = pdf(gmm_model,xVals');
% figure;
% plot(xVals,yVals);
% hold on;
% cfPlot=plot(1,1);


for trialInd = 1:maxTrial
    % Common starting parameter
    %theta0 = normrnd(0,1,M,1);
    %theta0 = normrnd(2,1,M,1);
    theta0 = unifrnd(-4,4,M,1);
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
        


        % Iterations
        for iter = 1:maxIter
            % Get update
            timeStart = tic;
            
            modelOpt.nextSeed = trialInd * maxIter + iter;
            [theta, gradInfo, modelOpt] = svgd_singleIteration(theta, dlog_p, gradInfo, modelOpt);
            
%             if modelInd == 3
%                 pause(0.2);
%                 delete(cfPlot);
%                 cfPlot = plot(theta,ones(1,50)*0.1,'o');
%                 %cfPlot = plot(gradInfo.ksdInfo.y,gradInfo.ksdInfo.w,'o');
%                 drawnow;
%             end
            
            timePassed = timePassed + toc(timeStart);

            if mod(iter, timeStepUnit) == 0 % Print and evaluate at current step
                [MSE_x, MSE_xsq] = evalPerformance(theta);

                iterUnit = iter / timeStepUnit;
                fprintf('Evaluating (%d/%d trials) : Model (%d/%d) - Iteration (%d/%d)\n', ...
                        trialInd, maxTrial, modelInd, numModels, iter, maxIter);
                t_vals(modelInd,iterUnit) = t_vals(modelInd,iterUnit) + timePassed / maxTrial;
                mse_x(modelInd,iterUnit) = mse_x(modelInd,iterUnit) + MSE_x / maxTrial;
                mse_xsq(modelInd,iterUnit) = mse_xsq(modelInd,iterUnit) + MSE_xsq / maxTrial;
            end
        end
    end
end



%% Plot results
results = {t_vals,mse_x,mse_xsq};
%% Plot result

figure;
colOpts = {'h-','o-','*-','.-','x-','s-','d-','^-','v-','p-','h-','>-','<-'};
titleNames = {'Total Time', 'E[x]', 'E[x^2]'};
valNames = {'log10 t','log10 MSE','log10 MSE'};
plotTime = timeStepUnit * (1:numTimeSteps);

for j = 1:3
    subplot(1,4,j);
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

subplot(1,4,4);
axis off;
leg1 = legend(handles,algNames, 'Orientation','vertical');
%set(leg1, 'Position',[0.7 0.3 0 0]);
set(leg1, 'Position',[0.8 0.5 0.05 0.05]);






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

function [MSE_x, MSE_xsq] = evalPoints(points, true_x, true_xsq)
    x_hat = mean(points,1);
    xsq_hat = mean(points.^2,1);

    MSE_x = log10(MSE(x_hat, true_x));
    MSE_xsq = log10(MSE(xsq_hat, true_xsq));
end

function val = MSE(y,yhat)
    val = mean((y(:) - yhat(:)).^2);
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
