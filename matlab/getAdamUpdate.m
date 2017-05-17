function [ adam_grad, hist_grad ] = getAdamUpdate(ori_grad, hist_grad, stepsize, auto_corr, fudge_factor)
% Returns the update to perform using ADAM
% --INPUTS
%   - ori_grad : original gradient
%   - hist_grad : historical gradient
%   - stepsize : stepsize of the update
%   - auto_corr : alpha parameter of adam
%   - fudge_factor : fudge factor

% -- OUTPUTS
%   - adam_grad : gradient step using adam
%   - hist_grad : updated hist_grad
    if hist_grad == 0
        hist_grad = hist_grad + ori_grad.^2;
    else
        hist_grad = auto_corr * hist_grad + (1-auto_corr) * ori_grad.^2;
    end
    adj_grad = ori_grad ./ (fudge_factor + sqrt(hist_grad));
    adam_grad = stepsize * adj_grad;

end

