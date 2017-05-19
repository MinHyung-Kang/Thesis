function [ logp, grad ] = nuts_lr_binary( nlogpdf, theta)

    [logp, grad] = nlogpdf(theta);
    logp = -logp;
    grad = -grad';
end
%{
function [ logp, grad ] = nuts_lr_logp( data, sigma, theta)
%% a function that returns the log probability its gradient evaluated at theta.


[n, d] = size(data.X);

% log value
logp = -sum( log_exp( -data.Y.*sum(repmat(theta,n,1).*data.X, 2) ) ) - sum(theta.^2)/(2*sigma^2);
% gradient of log probability
%grad =  sum( ( repmat(expval,1,d) .* (-repmat(data.Y,1,d).*data.X) ) ./ repmat((1 + expval),1,d) , 1) - theta/(sigma^2);
expval = exp(data.Y.*sum(repmat(theta,n,1).*data.X, 2));
grad = sum( (-repmat(data.Y,1,d).*data.X) ./ repmat((1 + expval),1,d) , 1) - theta/(sigma^2);
end
%}
