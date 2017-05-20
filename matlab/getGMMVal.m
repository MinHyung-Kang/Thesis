% Creates a function handle that returns the values of logp and gradp for
% given dataset
function [logp, gradp] = getGMMVal(pdistrib,X)
    [N, T] = size(X);
    pVal = pdf(pdistrib,X);
    logp = log(pVal);

    gradp = gmm_dlogp(pdistrib, X)
end
