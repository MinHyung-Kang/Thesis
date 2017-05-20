% Compares the results with that of NUTS sampler
% NUTS: example http://arxiv.org/abs/1111.4246

clear
rng(1);

addpath('./nuts/')

% parameters for NUTS
n_warm_up = 10000;

N = 10000; % number of samples to generate
a0 = 1; b0 = 1; % hyper-parameters

%% SVGD Options
MOpts = [10,20,30,50,80,100,150,200,250];
mOpts = [5,10,10,20,20,20,35,40,50];
maxIter = 3000;  % maximum iteration times
numTimeSteps = 10; % How many timesteps we want to shows
maxTrial = 10;% How many trials we want to average over
timeStepUnit = maxIter / numTimeSteps;

adverIter = 10;
optNum = length(MOpts);
algNames = {'SVGD','Random Subset', 'Random Subset + Control Functional', ...
    'Induced Points', 'Adversarial Induced Points (10 updates)'};
numModels = length(algNames);
baseModel = getModel('none',-1);                                   % Base Model

%% NUTS Sampling

dimOpts = [10,25,50,100];
kOpts = [10,25,50,100];

%dimOpts = [5,10];
%kOpts = [5,10];


for dimInd = 1:length(dimOpts)
    d = dimOpts(dimInd);
    for kInd = 1:length(kOpts)

        k = kOpts(kInd);
        mu = k * randn(k,d);            % Mean
        sigma = repmat(eye(d),1,1,k);   % Covariance
        for i = 1:k
            Crand = rand(d,d);
            sigma(:,:,i) = (Crand + Crand' + d * eye(d))/d;
        end
        w = rand(k,1);  % Weights
        w = w/sum(w);
        pdistrib = gmdistribution(mu, sigma, w');

        taskName = sprintf('%s_%ddim_%dclusters_%d_%d.mat','gmm',d,k,n_warm_up,N);

        % Log likelihood and derivative for nuts
        log_gmm_nuts = @(theta)getGMMVal(pdistrib, theta);

        % Derivative function handle for svgd
        dlog_p = @(theta,seed)gmm_dlogp(pdistrib, theta);

        % Use NUTS to get samples
        theta_nuts = random(pdistrib,N);

        x_nuts = mean(theta_nuts,1);
        cov_nuts = cov(theta_nuts);

        evalPerformance = @(points)evalPoints(points, x_nuts, cov_nuts);
        evalMMD = @(points)evalPointsMMD(points, theta_nuts);

        mse_x = zeros(numModels,optNum);
        mse_cov = zeros(numModels,optNum);
        t_vals = zeros(numModels, optNum);
        mmd_stat = zeros(numModels, optNum);

        % for m
        for mInd = 1:optNum
            M = MOpts(mInd);
            m = mOpts(mInd);

            % Redefine all the models for given m
            subsetModel = getModel('subset',-1, m);                            % Random Subset
            subsetCFModel = getModel('subsetCF',-1, m);                        % Random Subset (CF)
            inducedModel = getModel('inducedPoints',-1, m,0,-1,0,false);       % Induced Points
            inducedAdverModel = getModel('inducedPoints',-1, m,3,adverIter,0.1,true);   % Induced Points - update y,h

            modelOpts = {baseModel, subsetModel, subsetCFModel, inducedModel, inducedAdverModel};

            for trialInd = 1:maxTrial
                theta0 = randn(M,d);

                for modelInd = 1:length(modelOpts)
                    fprintf('[%d dimension /%d clusters]Evaluating : (M=%d) Trial (%d/%d) Model : %s \n',...
                        d, k, M, trialInd, maxTrial, algNames{modelInd});
                    modelOpt = modelOpts{modelInd};

                    % SVGD update
                    timeStart = tic;
                    modelOpt.baseSeed = trialInd * maxIter;
                    theta = svgd(theta0, dlog_p, maxIter, modelOpt);
                    timePassed = toc(timeStart);

                    [MSE_x, MSE_cov] = evalPerformance(theta);
                    HInfo= evalMMD(theta);

                    mse_x(modelInd, mInd) = mse_x(modelInd, mInd) + MSE_x / maxTrial;
                    mse_cov(modelInd, mInd) = mse_cov(modelInd, mInd) + MSE_cov / maxTrial;
                    t_vals(modelInd, mInd) = t_vals(modelInd, mInd) + timePassed/maxTrial;
                    mmd_stat(modelInd, mInd) = mmd_stat(modelInd, mInd) + HInfo.mmd.val/maxTrial;
                end
            end
        end

        results = {t_vals, mse_x, mse_cov, mmd_stat};
        matName = sprintf('./results/gmm_nuts_results_%s',taskName);
        save(matName,'results');

    end
end

return;

%% Plot the results

figure;
optNum = 9;
algNames = {'SVGD','Random Subset', 'Random Subset + Control Functional', ...
    'Induced Points', 'Adversarial Induced Points (10 updates)'};
numModles = length(algNames);
colOpts = {'h-','o-','*-','.-','x-','s-','d-','^-','v-','p-','h-','>-','<-'};
titleNames = {'Total Time', 'Estimating mean', 'Estimating covariance', 'Maximum Mean Discrepancy'};
yLabels = {'log10 t', 'log10 MSE', 'log10 MSE', 'test statistic'};

MOptsTxt = {'10','20','30','50','80','100','150','200','250'};


for j = 1:4
    if j >= 3
        subplot(2,3,j+1);
    else
        subplot(2,3,j);
    end

    handles = zeros(1, numModels);
    result = results{j};
    for i = 1:numModels
        if j == 1
            handles(i) = semilogy(1:optNum, result(i,:),colOpts{i});
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

subplot(2,3,6);
axis off;
leg1 = legend(handles, algNames, 'Orientation','vertical');
set(leg1, 'Position',[0.8 0.5 0.05 0.05]);


%% Auxiliary functions
function [MSE_x, MSE_xsq] = evalPoints(points, x_nuts, cov_nuts)
     x_hat = mean(points,1);
     cov_hat = cov(points);

     MSE = @(y,yhat)(mean((y(:) - yhat(:)).^2));

     MSE_x = log10(MSE(x_hat, x_nuts));
     MSE_xsq = log10(MSE(cov_hat, cov_nuts));
 end

 function [HInfo]= evalPointsMMD(points, points_nuts)
     points_total = [points;points_nuts];
     labels = [ones(1,size(points_nuts,1)) ones(1,size(points,1)) * -1];
     [~,HInfo]= mmd(points_total, labels);
 end


 % Creates a function handle that returns the values of logp and gradp for
 % given dataset
 function [logp, gradp] = getGMMVal(pdistrib,X)
     pVal = pdf(pdistrib,X);
     logp = log(pVal);

     gradp = gmm_dlogp(pdistrib, X, pVal);
 end

 function gradp = gmm_dlogp(pdistrib, X, pVal)
     mu = pdistrib.mu;
     sigma = pdistrib.Sigma;
     numComp = pdistrib.NumComponents;
     w = pdistrib.ComponentProportion;
     [N,T] = size(X);
     gradp = zeros(N, T);
     for k = 1:numComp % For each component
         mu_k = mu(k,:);
         covMat_i = sigma(:,:,k);
         w_k = w(k);
         front = mvnpdf(X, mu_k, covMat_i);   % n by 1
         back = (bsxfun(@minus, X, mu(k,:)) * (inv(covMat_i))'); % n by d
         gradp = gradp + (-1) * w_k * bsxfun(@times, front, back);
     end

     if nargin == 2
         pVal = log(pdf(pdistrib,X));
     end

     gradp = bsxfun(@rdivide, gradp, pVal);
 end
