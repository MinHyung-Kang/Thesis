%% Demo code to show mean/variance estimation of SVGD
%% Options

fudge_factor = 1e-6;

maxIter = 1000;  % maximum iteration times
maxTrial = 10;% How many trials we want to average over
adverIter = 10;
MOpts = [10,20,50,100,200,250,500,750,1000,1500,2000,2500,5000];
%mOpts = round(MOpts/5);
optNum = length(MOpts);
mOpts = ones(1,optNum) * 5; 
algNames = {'SVGD','Random Subset', 'Random Subset + Control Functional', ...
'Induced Points', 'Adversarial Induced Points(1 iteration)',...
'Adversarial Induced Points(Batch, 1 iteration)',...
'Adversarial Induced Points(5 iteration)', ...
'Adversarial Induced Points(Batch, 5 iteration)'};
modelNum = length(algNames);
% Score function (simplest possible)
dlog_p = @(X)(X);


%% Run iterations

baseModel = getModel('none',-1);                                   % Base Model

t_vals = zeros(modelNum, optNum);

for mInd = 1:optNum
    M = MOpts(mInd);
    m = mOpts(mInd);

    % Redefine all the models for given m
    subsetModel = getModel('subset',-1, m);                            % Random Subset
    subsetCFModel = getModel('subsetCF',-1, m);                        % Random Subset (CF)
    inducedModel = getModel('inducedPoints',-1, m,0,-1,0,false);       % Induced Points
    inducedAdverModel1 = getModel('inducedPoints',-1, m,3,1,0.1,true);   % Induced Points - update y,h
    inducedAdverModelSubset1 = getModel('inducedPoints',-1, m,3,1,0.1,true,m);   % Induced Points - update y,h
    inducedAdverModel5 = getModel('inducedPoints',-1, m,3,5,0.1,true);   % Induced Points - update y,h
    inducedAdverModelSubset5 = getModel('inducedPoints',-1, m,3,5,0.1,true,m);   % Induced Points - update y,h

    modelOpts = {baseModel, subsetModel, subsetCFModel, inducedModel, inducedAdverModel};

    for trialInd = 1:maxTrial
        % Common starting parameter
        theta0 = normrnd(0,1,M,1);

        for modelInd = 1:length(modelOpts)
            fprintf('Evaluating : (M=%d) Trial (%d/%d) Model : %s \n',...
                M, trialInd, maxTrial, algNames{modelInd});
            
            modelOpt = modelOpts{modelInd};
            modelOpt.baseSeed = trialInd * maxIter;
            % Get update
            timeStart = tic;
            theta = svgd(theta0, dlog_p, maxIter, modelOpt);
            timePassed = toc(timeStart);
            t_vals(modelInd, mInd) = t_vals(modelInd, mInd) + timePassed/maxTrial;

            save('allModels_t_vals.mat','t_vals');
        	indices = [mInd, trialInd, modelInd];
        	save('allModels_indices.mat','indices');
        end

    end
end


%% Plot results
figure;
colOpts = {'h-','o-','*-','.-','x-','s-','d-','^-','v-','p-','h-','>-','<-'};
titleNames = {'Total Time'};
yLabels = {'log10 t'};

MOptsTxt = ['10','20','50','100','200','250','500','750','1000','1500','2000'];
%MOptsTxt = {'10','20','50','100','200','250','500','750','1000'};
numModels = 5;


handles = zeros(1, numModels);
for i = 1:numModels
    handles(i) = semilogy(1:optNum, t_vals(i,1:optNum),colOpts{i});
    hold on;
end
title(sprintf('%s',titleNames{1}));
xlabel('Sample Size (N)');
ylabel(yLabels{1});
set(gca,'Xtick',[1 4 7 9 11]);
set(gca,'XtickLabel',{'10','100','500', '1000', '2000'});
leg1 = legend(handles, algNames, 'Orientation','vertical','Location','NorthWest');
%set(leg1, 'Position',[0.8 0.5 0.05 0.05]);
%%
%axis off;
%leg1 = 
%set(leg1, 'Position',[0.7 0.3 0 0]);

