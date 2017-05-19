%% Demo code to show mean/variance estimation of SVGD


%% Generate a random GMM
rng(1);
clear;

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

maxIter = 1000;  % maximum iteration times
maxTrial = 1;% How many trials we want to average over
adverIter = 10;
MOpts = [10,20,30,50,80,100,150,200,250];
optNum = length(MOpts);
mOpts = [5 ones(1,4) * 10  ones(1,4) * 20];
%mOpts2 = round(MOpts/4);
%mOpts2 = ones(1,9)*10;
mOpts2 = MOpts/5;
mOpts = mOpts2;
algNames = {'SVGD','Random Subset', 'Random Subset + Control Functional', ...
    'Induced Points', 'Adversarial Induced Points (10 updates)', 'Monte Carlo'};
%    'Induced Points', 'Monte Carlo'};

numModels = length(algNames);

%% Run iterations

baseModel = getModel('none',-1);                                   % Base Model

mse_x = zeros(numModels,optNum);
mse_xsq = zeros(numModels,optNum);
t_vals = zeros(numModels, optNum);

for mInd = 1:optNum
    M = MOpts(mInd);
    m = mOpts(mInd); m2 = mOpts2(mInd);

    % Redefine all the models for given m
    subsetModel = getModel('subset',-1, m);                            % Random Subset
    subsetCFModel = getModel('subsetCF',-1, m);                        % Random Subset (CF)
    inducedModel = getModel('inducedPoints',-1, m2,0,-1,0,false);       % Induced Points
    inducedAdverModel = getModel('inducedPoints',-1, m2,3,adverIter,0.1,true);   % Induced Points - update y,h

    modelOpts = {baseModel, subsetModel, subsetCFModel, inducedModel, inducedAdverModel, 'MonteCarlo'};
    %modelOpts = {baseModel, subsetModel, subsetCFModel, inducedModel, 'MonteCarlo'};

    for trialInd = 1:maxTrial
        % Common starting parameter
        %theta0 = normrnd(0,1,M,1);
       % theta0 = normrnd(-10,1,M,1);
        theta0 = unifrnd(-4,4,M,1);

        for modelInd = 1:length(modelOpts)
            fprintf('Evaluating : (M=%d) Trial (%d/%d) Model : %s \n',...
                M, trialInd, maxTrial, algNames{modelInd});

            modelOpt = modelOpts{modelInd};
            
            % Get update
            timeStart = tic;
            if modelInd == length(modelOpts)
                theta = random(gmm_model, M);
            else
                modelOpt.baseSeed = trialInd * maxIter;
                theta = svgd(theta0, dlog_p, maxIter, modelOpt);
            end
            timePassed = toc(timeStart);

            [MSE_x, MSE_xsq] = evalPerformance(theta);

            mse_x(modelInd, mInd) = mse_x(modelInd, mInd) + MSE_x / maxTrial;
            mse_xsq(modelInd, mInd) = mse_xsq(modelInd, mInd) + MSE_xsq / maxTrial;
            t_vals(modelInd, mInd) = t_vals(modelInd, mInd) + timePassed/maxTrial;
        end

    end
end


%% Plot results
results = {t_vals,mse_x,mse_xsq};
figure;
colOpts = {'h-','o-','*-','.-','x-','s-','d-','^-','v-','p-','h-','>-','<-'};
titleNames = {'Total Time', 'Estimating x', 'Estimating x^2'};
yLabels = {'log10 t', 'log10 MSE', 'log10 MSE'};

MOptsTxt = {'10','20','30','50','80','100','150','200','250'};


for j = 1:3
    subplot(1,4,j);
    handles = zeros(1, numModels);
    result = results{j};
    for i = 1:numModels
        if j == 1
            if i ~= numModels
                handles(i) = semilogy(1:optNum, result(i,:),colOpts{i});
            end
        else
            handles(i) = plot(1:optNum, result(i,:),colOpts{i});
        end
        hold on;
    end
    title(sprintf('%s',titleNames{j}));
    xlabel('Sample Size (N)');
    ylabel(yLabels{j});
    set(gca,'Xtick',[1 4 9]);
    set(gca,'XtickLabel',{'10','50','250'});
end

subplot(1,4,4);
axis off;
leg1 = legend(handles, algNames, 'Orientation','vertical');
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
