clear

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sample code to test code on bayesian logistic regression
% Implemented from Liu, Q. and Wang, D. (2016) Stein Variational Gradient Descent
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

M = 200; % number of particles

% we partition the data into 80% for training and 20% for testing
train_ratio = 0.8;

% build up training and testing dataset
load ../data/covertype.mat;
X = covtype(:,2:end); y = covtype(:,1); y(y==2) = -1;

X = [X, ones(size(X,1),1)];  % the bias parameter is absorbed by including 1 as an entry in x
[N, d] = size(X); D = d+1; % w and alpha (prameters)

% building training and testing dataset
train_idx = randperm(N, round(train_ratio*N));  test_idx = setdiff(1:N, train_idx);
X_train = X(train_idx, :); y_train = y(train_idx);
X_test = X(test_idx, :); y_test = y(test_idx);

n_train = length(train_idx); n_test = length(test_idx);

% example of bayesian logistic regression
batchsize = 100; % subsampled mini-batch size
a0 = 1; b0 = .01; % hyper-parameters

dlog_p  = @(theta,seed)dlog_p_lr(theta, X_train, y_train,batchsize,seed,a0,b0); % returns the first order derivative of the posterior distribution


%% Define models
m = 40;
adverIter = 10;

% Common options
baseModel = getModel('none',-1);                                   % Base Model
subsetModel = getModel('subset',-1, m);                            % Random Subset
subsetCFModel = getModel('subsetCF',-1, m);                        % Random Subset (CF)
inducedModel = getModel('inducedPoints',-1, m,0);                  % Induced Points
inducedAdver_yModel = getModel('inducedPoints',-1, m,1,adverIter); % Induced Points - update y
inducedAdver_hModel = getModel('inducedPoints',-1, m,2,adverIter); % Induced Points - update h
inducedAdverModel = getModel('inducedPoints',-1, m,3,adverIter,0.1,true);   % Induced Points - update y,h
inducedAdverModelBatch = getModel('inducedPoints',-1, m,3,adverIter,0.1,true,m);   % Induced Points - update y,h (batch)

% Induced Points - update y,h; no Additional methods used
inducedAdverModel_basic = getModel('inducedPoints',-1, m,3,adverIter,0,false);
% Induced Points - update y,h; use uStats
inducedAdverModel_justU = getModel('inducedPoints',-1, m,3,adverIter,0,true);
% Induced Points - update y,h; use regularization
inducedAdverModel_justReg = getModel('inducedPoints',-1, m,3,adverIter,0.1,false);
inducedAdverModel_justReg2 = getModel('inducedPoints',-1, m,3,adverIter,0.2,false);
inducedAdverModel_justReg3 = getModel('inducedPoints',-1, m,3,adverIter,0.05,false);

%% Compare options
maxIter = 4000;  % maximum iteration times
numTimeSteps = 10; % How many timesteps we want to shows
maxTrial = 10;% How many trials we want to average over

timeStepUnit = maxIter / numTimeSteps;
%1. Compare subset models
% modelOpts = {baseModel, subsetModel, subsetCFModel};
% algNames= {'Base','subset','subset(CF)'};

%modelOpts = {baseModel, inducedModel, inducedAdverModel_basic,inducedAdverModel};
%algNames= {'Base','inducedPoints(basic)','inducedPoints(Adverse)','inducedPoints(Adverse-uStat/reg=0.1)'};

% 2. Compare Models of Basic, inducedPoints, inducedPoints(5Updates),
% inducedPoints(10Updates)

% inducedAdverModel5 = getModel('inducedPoints',-1, m,3,5,0.1,false);   % Induced Points - update y,h
% inducedAdverModel10 = getModel('inducedPoints',-1, m,3,10,0.1,false);   % Induced Points - update y,h
% modelOpts = {baseModel, inducedModel,inducedAdverModel5, inducedAdverModel10};
% algNames= {'base','IP','AIP(10 iter)','AIP(50 iter)'};

modelOpts = {baseModel, inducedModel, inducedAdverModel, inducedAdverModelBatch};
algNames= {'SVGD','Induced Points','Adversarial Induced Points','Adversarial Induced Points (Batch)'};


numModels = length(modelOpts);   % How many models we want to try

%% Perform SVGD
tVal = zeros(numModels, numTimeSteps);
acc = zeros(numModels, numTimeSteps);
llh = zeros(numModels, numTimeSteps);
valNames = {'t','acc','llh'};

for trialInd = 1:maxTrial
    % Common starting parameters
    alpha0 = gamrnd(a0, b0, M, 1);
    theta0 = zeros(M, D);
    for i = 1:M
        theta0(i,:) = [normrnd(0, sqrt((1/alpha0(i))), 1, d), log(alpha0(i))]; % w and log(alpha)
    end

    currentSeed = trialInd;

    for modelInd = 1:numModels
        theta = theta0;
        modelOpt = modelOpts{modelInd};
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
            timePassed = timePassed + toc(timeStart);

            if mod(iter, timeStepUnit) == 0 % Print and evaluate at current step
                [acc_svgd, llh_svgd] = bayeslr_evaluation(theta, X_test, y_test);

                iterUnit = iter / timeStepUnit;
                fprintf('Evaluating (%d/%d trials) : Model (%d/%d) - Iteration (%d/%d)\n', ...
                        trialInd, maxTrial, modelInd, numModels, iter, maxIter);
                tVal(modelInd,iterUnit) = tVal(modelInd,iterUnit) + timePassed / maxTrial;
                acc(modelInd,iterUnit) = acc(modelInd,iterUnit) + acc_svgd / maxTrial;
                llh(modelInd,iterUnit) = llh(modelInd,iterUnit) + llh_svgd / maxTrial;
            end
        end

    end
end

results = struct(valNames{1},tVal,valNames{2},acc,valNames{3}, llh);

%% Plot result

figure('Position', [100, 100, 800, 600]);
valNames = {'t','acc','llh'};
colOpts = {'h-','o-','*-','.-','x-','s-','d-','^-','v-','p-','h-','>-','<-'};
titleNames = {'Total Time', 'Testing Accuracy', 'Testing Log-Likelihood'};
yAxisNames = {'log10 t', 'accuracy', 'log-likelihood'};
plotTime = timeStepUnit * (1:numTimeSteps);

for j = 1:3
    subplot(2,2,j);
    handles = zeros(1, numModels);
    result = results.(valNames{j});

    for i = 1:numModels
        if j == 1
            handles(i) = semilogy(plotTime, result(i,:),colOpts{i},'LineWidth',1.5);
        else
            handles(i) = plot(plotTime, result(i,:),colOpts{i},'LineWidth',1.5);
        end
        hold on;
       %handles(i) = plot(plotTime, result(i,:),colOpts{i});
    end
    set(gca,'FontSize',15);
    title(sprintf('%s',titleNames{j}),'FontSize',16);
    xlabel('Iterations','FontSize',16);
    ylabel(yAxisNames{j},'FontSize',16);

end

subplot(2,2,4);
axis off;
leg1 = legend(handles,algNames, 'Orientation','vertical');
%set(leg1, 'Position',[0.7 0.3 0 0]);
set(leg1, 'Position',[0.7 0.3 0.05 0.05], 'FontSize',12);
