function y_hat = fClassify_LogisticReg(X_test, theta)
% This function returns the probability for each pattern of the test set to
% belong to the positive class using the logistic regression classifier.
%
% INPUT
%   - X_test: Matrix with dimension (m_t x n) with the test data, where m_t
%   is the number of test patterns and n is the number of features (i.e. 
%   the length of the feature vector that define each element). 
%   - theta: Vector with length n (i.e., the number of features of each 
%   pattern along with the parameters theta of the estimated h function
%   after the training.
%   
% OUTPUT
%	- t
%   y_hat: Vector of length m_t with the estimations made for each test
%   element by means of the logistic regression classifier. These
%   estimations correspond to the probabilities that these elements belong
%   to the positive class.
%

    numElemTest = size(X_test,1);
    y_hat = zeros(1, numElemTest);
    
    for i=1:numElemTest
        x_test_i = [1, X_test(i,:)]; % Put a 1 (value for x0) at the beginning of each pattern
        % ====================== YOUR CODE HERE ======================
        y_hat(i) = fun_sigmoid(theta, x_test_i); 
        % ============================================================
    end
end