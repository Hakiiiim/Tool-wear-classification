function cost_i = fCalculateCostLogReg(y, y_hat)
% Calculates the cost of the OUTPUT OF JUST ONE pattern from the logistic 
% regression classifier (i.e. the result of applying the h function) and 
% its real class. 
%
% INPUT
%   - y: Real class.
%   - y_hat: Output of the h function (i.e. the hypothesis of the logistic
%   regression classifier.
%
% OUTPUT
%   cost_i: Escalar with the value of the cost of the estimated output
%   y_hat.
%

% ====================== YOUR CODE HERE ======================
cost_i = y*log(y_hat)+(1-y)*log(1-y_hat);
% ============================================================
    
end