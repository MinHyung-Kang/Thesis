%% Demo code to show mean/variance estimation of SVGD


%% Generate a random GMM
rng(1);
clear;

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


%% Options

fudge_factor = 1e-6;

maxIter = 2000;  % maximum iteration times
maxTrial = 1;% How many trials we want to average over
adverIter = 10;
MOpts = [50,80,100,150,200];
optNum = length(MOpts);
mOpts2 = [10,15,20,30,40];
mOpts = mOpts2;
algNames = {'SVGD','Induced Points', 'Adversarial Induced Points (10 updates)'};
%    'Induced Points', 'Monte Carlo'};

numModels = length(algNames);

%% Run iterations

baseModel = getModel('none',-1);                                   % Base Model

acc = zeros(numModels,optNum);
llh = zeros(numModels,optNum);
tVal = zeros(numModels, optNum);

for mInd = 1:optNum
    M = MOpts(mInd);
    m = mOpts(mInd); m2 = mOpts2(mInd);

    % Redefine all the models for given m
    subsetModel = getModel('subset',-1, m);                            % Random Subset
    subsetCFModel = getModel('subsetCF',-1, m);                        % Random Subset (CF)
    inducedModel = getModel('inducedPoints',-1, m2,0,-1,0,false);       % Induced Points
    inducedAdverModel = getModel('inducedPoints',-1, m2,3,adverIter,0.1,true);   % Induced Points - update y,h

    %modelOpts = {baseModel, subsetModel, subsetCFModel, inducedModel, inducedAdverModel, 'MonteCarlo'};
    modelOpts = {baseModel, inducedModel,  inducedAdverModel};

    for trialInd = 1:maxTrial
        % Common starting parameters
        alpha0 = gamrnd(a0, b0, M, 1);
        theta0 = zeros(M, D);
        for i = 1:M
            theta0(i,:) = [normrnd(0, sqrt((1/alpha0(i))), 1, d), log(alpha0(i))]; % w and log(alpha)
        end

        currentSeed = trialInd;

        for modelInd = 1:length(modelOpts)
            fprintf('Evaluating : (M=%d) Trial (%d/%d) Model : %s \n',...
                M, trialInd, maxTrial, algNames{modelInd});

            modelOpt = modelOpts{modelInd};

            % Get update
            timeStart = tic;

            modelOpt.baseSeed = trialInd * maxIter;
            theta = svgd(theta0, dlog_p, maxIter, modelOpt);

            timePassed = toc(timeStart);

            [acc_svgd, llh_svgd] = bayeslr_evaluation(theta, X_test, y_test);

            fprintf('Evaluating (%d/%d trials) : Model (%d/%d)\n', ...
                trialInd, maxTrial, modelInd, numModels);
            tVal(modelInd,mInd) = tVal(modelInd,mInd) + timePassed / maxTrial;
            acc(modelInd,mInd) = acc(modelInd,mInd) + acc_svgd / maxTrial;
            llh(modelInd,mInd) = llh(modelInd,mInd) + llh_svgd / maxTrial;
        end

    end
end


%% Plot results
results = {tVal,acc,llh};
figure;
colOpts = {'h-','o-','*-','.-','x-','s-','d-','^-','v-','p-','h-','>-','<-'};
titleNames = {'Total Time', 'Accuracy', 'Log Likelihood'};
yLabels = {'log10 t', 'acc', 'llh'};

MOptsTxt = {'50','80','100','150','200'};



for j = 1:3
    subplot(1,4,j);
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

subplot(1,4,4);
axis off;
leg1 = legend(handles, algNames, 'Orientation','vertical');
%set(leg1, 'Position',[0.7 0.3 0 0]);
set(leg1, 'Position',[0.8 0.5 0.05 0.05]);
