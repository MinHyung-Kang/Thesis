function [ opt ] = getModel(method, h, m, adver, adverIter, alpha, uStat, batchSize)
% Function that returns a given model
% Input:
% -- method : Name of the method to use
%       ['none','subset','subsetCF','inducedPoints']
% -- h : bandwidth of kernel
% -- m : size of subset or number of induced particles to be used
% -- adver : whether to use adversarial update
%       [0 : do not use, 1 : just y, 2: just h, 3: both y and h]
% -- adverIter : number of adverse iterations to take
% -- alpha : regularization parameter
% -- uStat : Whether to compute u-Statistics or not
% -- batchSize : take subset for M


    % Common options
    opts.master_stepsize = 0.1;
    opts.auto_corr = 0.9;
    opts.method = 'adagrad';
    opts.kernel_opts.h = -1;
    opts.kernel_opts.adver = 0;     % Whether to use adversairal update or not
    opts.kernel_opts.adverIter = 1; % Number of adverse iterations to take
    opts.kernel_opts.alpha = 0.1;   % Regularization paremeter
    opts.kernel_opts.uStat = true;  % Whether to compute uStats or not
    opts.kernel_opts.batchSize = -1;
    opt = opts;


    opt.kernel_opts.method = method;

    % Set default bandwidth
    if nargin >= 2
        opt.kernel_opts.h = h;
    end

    % Set other options
    if strcmp(method,'none') ~= 1
        if nargin < 3
            error('Require at least 3 inputs');
        else
            opt.kernel_opts.m = m;
        end

        if nargin >= 4
            opt.kernel_opts.adver = adver;
        end

        if nargin >=5
            opt.kernel_opts.adverIter = adverIter;
        end

        if nargin >= 6
            opt.kernel_opts.alpha = alpha;
            opt.kernel_opts.uStat = uStat;
        end
        
        if nargin >= 8
            opt.kernel_opts.batchSize = batchSize;
        end

    end

end
