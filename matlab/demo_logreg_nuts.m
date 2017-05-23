% Compares the results with that of NUTS sampler
% NUTS: example http://arxiv.org/abs/1111.4246

clear

%% NUTS Options
% test on 13 binary classification datasets
load benchmarks.mat;
addpath('./nuts/')

% parameters for NUTS
n_warm_up = 1000;
train_ratio = 0.8;

N = 10000; % number of samples to generate
a0 = 1; b0 = 1; % hyper-parameters


%% SVGD Options
MOpts = [10,20,30,50,80,100,150,200,250];
%MOpts = [10,20];
mOpts = [5,10,10,20,20,20,35,40,50];
%mOpts = [5 ones(1,4) * 10  ones(1,4) * 20];
maxIter = 3000;  % maximum iteration times
numTimeSteps = 10; % How many timesteps we want to shows
maxTrial = 10;% How many trials we want to average over
timeStepUnit = maxIter / numTimeSteps;
%m = 20;
adverIter = 10;
optNum = length(MOpts);
algNames = {'SVGD','Random Subset', 'Random Subset + Control Functional', ...
    'Induced Points', 'Adversarial Induced Points (10 updates)'};
%    'Induced Points'};
numModels = length(algNames);
baseModel = getModel('none',-1);                                   % Base Model

%% NUTS Sampling
for datasetInd = 7:length(benchmarks)
    dataset = benchmarks{datasetInd};
    % If saved, load that dataset
    datasetName = sprintf('%s_%d_%d.mat', dataset,n_warm_up,N);
    if exist(datasetName,'file') > 0
        load(datasetName);
        [~, d] = size(Xtrain);
        D = d+1;
    else
        bm = eval(char(dataset));
        X = bm.x; Y = bm.t;

        X = [ones(size(X,1),1), X]; %bias term
        [n, d] = size(X);

        %% random partition
        train_idx = randperm(n, round(train_ratio*n)); test_idx = setdiff(1:n, train_idx);
        Xtrain = X(train_idx, :); Ytrain = Y(train_idx);
        Xtest = X(test_idx, :); Ytest = Y(test_idx);
        train_num = length(train_idx);

        % training/testing
        D = d+1; % number of parameters (w & alpha)
        ntrain = size(Xtrain, 1); ntest = size(Xtest, 1);
        log_bayeslr_nuts = @(theta) logreg(theta, Xtrain, Ytrain, a0, b0); % log posterior

        % NUTS
        tic
            [theta_nuts, ~] = nuts_da(log_bayeslr_nuts, N, n_warm_up, randn(1, D));
        toc;
        % Save the results
        save(datasetName, 'theta_nuts', 'Xtrain', 'Ytrain');
    end
    x_nuts = mean(theta_nuts,1);
    cov_nuts = cov(theta_nuts);

    evalPerformance = @(points)evalPoints(points, x_nuts, cov_nuts);
    evalMMD = @(points)evalPointsMMD(points, theta_nuts);

    mse_x = zeros(numModels,optNum);
    mse_cov = zeros(numModels,optNum);
    t_vals = zeros(numModels, optNum);
    mmd_stat = zeros(numModels, optNum);

    %% SVGD
    dlog_p = @(theta,seed) dlog_p_lr(theta,Xtrain,Ytrain,size(Xtrain,1),seed,a0,b0);

    % For each option
    for mInd = 1:optNum
        M = MOpts(mInd);
        m = mOpts(mInd);

        % Redefine all the models for given m
        subsetModel = getModel('subset',-1, m);                            % Random Subset
        subsetCFModel = getModel('subsetCF',-1, m);                        % Random Subset (CF)
        inducedModel = getModel('inducedPoints',-1, m,0,-1,0,false);       % Induced Points
        inducedAdverModel = getModel('inducedPoints',-1, m,3,adverIter,0.1,true);   % Induced Points - update y,h

        modelOpts = {baseModel, subsetModel, subsetCFModel, inducedModel, inducedAdverModel};
        %modelOpts = {baseModel, subsetModel, subsetCFModel, inducedModel};

        % Try this many times
        for trialInd = 1:maxTrial

            % Initialize
            alpha0 = gamrnd(a0, b0, M, 1);
            theta0 = zeros(M, D);
            for i = 1:M
                theta0(i,:) = [normrnd(0, sqrt((1/alpha0(i))), 1, d), log(alpha0(i))]; % w and log(alpha)
            end

            % Evaluate for each model
            for modelInd = 1:length(modelOpts)
                fprintf('[%s]Evaluating : (M=%d) Trial (%d/%d) Model : %s \n',...
                    dataset, M, trialInd, maxTrial, algNames{modelInd});

                modelOpt = modelOpts{modelInd};
                modelOpt.baseSeed = trialInd * maxIter;

                % Get update
                timeStart = tic;

%                  for iter = 1:maxIter
%                     % Get update
%                     timeStart = tic;
%                     modelOpt.nextSeed = trialInd * maxIter + iter;
%                     [theta, gradInfo, modelOpt] = svgd_singleIteration(theta, dlog_p, gradInfo, modelOpt);
%                  end

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
    matName = sprintf('logreg_nuts_results_%s',datasetName);
    save(matName,'results');    
end


return;  % EARLY STOP HERE
%% Plot the results (only mmd and time)
figure('Position', [100, 100, 800, 250]);
%figure;
optNum = 9;
algNames = {'SVGD','Random Subset', 'Random Subset + Control Functional', ...
    'Induced Points', 'Adversarial Induced Points (10 updates)'};
numModels = length(algNames);
colOpts = {'h-','o-','*-','.-','x-','s-','d-','^-','v-','p-','h-','>-','<-'};
titleNames = {'Total Time','Maximum Mean Discrepancy'};
yLabels = {'log10 t', 'test statistic'};
datasetName = 'German'

MOptsTxt = {'10','20','30','50','80','100','150','200','250'};

for j=1:2
    subplot(1,2,j);
    handles = zeros(1, numModels);
    if j == 2
        resultInd = 4;
    else
        resultInd = 1;
    end
    result = results{resultInd};
    
    for i = 1:numModels
        if j == 1
            handles(i) = semilogy(1:optNum, result(i,:),colOpts{i});
        else
            handles(i) = plot(1:optNum, result(i,:),colOpts{i});
        end
        hold on;
    end
    %title(sprintf('%s',titleNames{j}),'FontSize',14);
    xlabel('Sample Size (N)','FontSize',14);
    ylabel(yLabels{j},'FontSize',14);
    set(gca,'Xtick',[1 4 9]);
    set(gca,'XtickLabel',{'10','50','250'});
end


%subplot(1,2,3);
%axis off;
% leg1 = legend(handles, algNames, 'Orientation','horizontal');
% %set(leg1, 'Position',[0.7 0.3 0 0]);
% set(leg1, 'Position',[0.4 0.1 0.05 0.05]);

ax1 = axes('Position',[0 0 1 1],'Visible','off');
axes(ax1) % sets ax1 to current axes
fileName = text(.025,0.38,datasetName);
fileName.FontSize = 18;
set(fileName, 'rotation', 90)



% %% Plot the results (all)
% figure;
% optNum = 9;
% algNames = {'SVGD','Random Subset', 'Random Subset + Control Functional', ...
%     'Induced Points', 'Adversarial Induced Points (10 updates)'};
% numModels = length(algNames);
% colOpts = {'h-','o-','*-','.-','x-','s-','d-','^-','v-','p-','h-','>-','<-'};
% titleNames = {'Total Time', 'Estimating mean', 'Estimating covariance', 'Maximum Mean Discrepancy'};
% yLabels = {'log10 t', 'log10 MSE', 'log10 MSE', 'test statistic'};
% 
% MOptsTxt = {'10','20','30','50','80','100','150','200','250'};
% 
% 
% for j = 1:4
%     if j >= 3
%         subplot(2,3,j+1);
%     else
%         subplot(2,3,j);
%     end
% 
%     handles = zeros(1, numModels);
%     result = results{j};
%     for i = 1:numModels
%         if j == 1
%             handles(i) = semilogy(1:optNum, result(i,:),colOpts{i});
%         else
%             handles(i) = plot(1:optNum, result(i,:),colOpts{i});
%         end
%         hold on;
%     end
%     title(sprintf('%s',titleNames{j}));
%     xlabel('Sample Size (N)');
%     ylabel(yLabels{j});
%     set(gca,'Xtick',[1 4 9]);
%     set(gca,'XtickLabel',{'10','50','250'});
% end
% 
% subplot(2,3,6);
% axis off;
% leg1 = legend(handles, algNames, 'Orientation','vertical');
% %set(leg1, 'Position',[0.7 0.3 0 0]);
% set(leg1, 'Position',[0.8 0.5 0.05 0.05]);

%% Auxiliary functions
% function [MSE_x, MSE_xsq] = evalPoints(points, x_nuts, cov_nuts)
%     x_hat = mean(points,1);
%     cov_hat = cov(points);
%
%     MSE = @(y,yhat)(mean((y(:) - yhat(:)).^2));
%
%     MSE_x = log10(MSE(x_hat, x_nuts));
%     MSE_xsq = log10(MSE(cov_hat, cov_nuts));
% end
%
% function [HInfo]= evalPointsMMD(points, points_nuts)
%     points_total = [points;points_nuts];
%     labels = [ones(1,size(points_nuts,1)) ones(1,size(points,1)) * -1];
%     [~,HInfo]= mmd(points_total, labels);
% end
