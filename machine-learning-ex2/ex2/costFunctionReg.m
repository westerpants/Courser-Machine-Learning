function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% X: 100x3
% y: 100x1
% theta: 3x1
% hx: 100x1

hx=sigmoid(X*theta);

J = (-y'*log(hx) - (1-y')*log(1-hx))/m + lambda * sum(theta(2:size(theta)).^2)/(2*m);

for i=1:m
  grad = grad + (hx(i)-y(i))*X(i,:)';
end
grad = grad / m;

for i=2:size(theta)
  grad = grad + lambda * theta(i) / m;
end

J
grad
% =============================================================

end
