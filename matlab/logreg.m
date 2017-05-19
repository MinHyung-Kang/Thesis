function [f, df] = logreg(theta, X, y, a0,b0)
% input:
% -- X: input data
% -- y: labels
% -- a0, b0: hyperparameters of gamma distribution
% output:
% -- f: log-likelihood
% -- df: first derivative

if nargin < 4; a0 = 1; end
if nargin < 5; b0 = 1; end

if size(theta,2)==1; theta = theta'; end
w = theta(1:end-1);
alpha = exp(theta(end));
D = length(w);

m = X*w';
p = 1./(1+exp(-y.*m));
av = (alpha/2)*(w*w');
loglik = sum(log(p));
logprior_w = (D/2)*log(alpha/(2*pi)) - av;
logprior_alpha = (a0-1).*log(alpha) - b0.*alpha;
f = loglik + logprior_w + logprior_alpha + theta(end);

% first derivatives
if nargout > 1
    c_hat = 1./(1+exp(-m));
    dw = ((y+1)/2 - c_hat)'*X - alpha*w;
    dalpha = D/2 - av + (a0-1) - b0.*alpha;
    dalpha = dalpha + 1;
    df = [dw dalpha];
end

