function g = fun_sigmoid(theta, X)
% This function calculates the sigmoid function g(z), where z is a linear
% combination of the parameters theta and the feature vector X's components
%
% INPUT
%	- theta: Parameters of the h function of the logistic regression
%	classifier.
% 	- X: Vector containing the data of one pattern.
%
% OUTPUT
%	g: Result of applying the sigmoid function using the linear combination
%	of theta and X.
%

% ====================== YOUR CODE HERE ======================
    ps = theta*transpose(X);
    g = 1/(1+exp(-1*ps));

% ============================================================
	
end