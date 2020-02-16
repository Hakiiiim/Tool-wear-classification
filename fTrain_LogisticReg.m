function theta = fTrain_LogisticReg(X_train, Y_train, alpha)
% This function implements the training of a logistic regression classifier
% using the training data (X_train) and its classes (Y_train).  
%
% INPUT
%   - X_train: Matrix with dimensions (m x n) with the training data, where
%   m is the number of training patterns (i.e. elements) and n is the
%   number of features (i.e. the length of the feature vector which
%   characterizes the object).
%   - Y_train: Vector that contains the classes of the training patterns.
%   Its length is m.
%   - alpha: Learning rate for the gradient descent algorithm.
%
% OUTPUT
%   theta: Vector with length n (i.e, the same length as the number of
%   features on each pattern). It contains the parameters theta of the
%   hypothesis function obtained after the training.
%

    % CONSTANTS
    % =================
    VERBOSE = true;
    max_iter = 1000; % Try with a different number of iterations
    % =================

    % Number of training patterns.
    m = size(X_train,1);

    % Allocate space for the outputs of the hypothesis function for each
    % training pattern
    h_train = zeros(1,m);
    
    % Allocate spaces for the values of the cost function on each iteration
    J = zeros(1, 1+max_iter);
    
    % Initialize the vector to store the parameters of the hypothesis
    % function
    theta = zeros(1, 1+size(X_train,2));

% *************************************************************************
% CALCULATE THE VALUE OF THE COST FUNCTION FOR THE INITIAL THETAS
    
    % THIS PIECE OF CODE IS JUST AN IDEA. YOU CAN DO IT AS YOU PREFER. 
    % CALL THE FUNCTIONS fun_sigmoidal and fCalculateCostLogReg
    %a. Intermediate result: Get the error for each element to sum it up.
    total_cost = 0;
    for i=1:m
        x_i = [1, X_train(i, :)]; % Put a 1 (value for x0) at the beginning of each pattern
        
        % Expected output (i.e. result of the sigmoid function) for i-th pattern
        % ====================== YOUR CODE HERE ======================
        h_train(i) = fun_sigmoid(theta, x_i);
        % ============================================================
        
        % Calculate the cost for the i-the pattern and add the cost of the last patterns
        % ====================== YOUR CODE HERE ======================
        total_cost = total_cost + fCalculateCostLogReg(Y_train(i), h_train(i));
        % ============================================================
    end
    
    % b. Calculate the total cost
    % ====================== YOUR CODE HERE ======================
    total_cost = (-1/m)*total_cost;
    J(1) = total_cost;
    % ============================================================

% *************************************************************************
% GRADIENT DESCENT ALGORITHM TO UPDATE THE THETAS

    % Iterative method carried out during a maximum number (max_iter) of iterations
    for num_iter=1:max_iter
        
        % *********************
        % STEP 1. Calculate the value of the function h with the current theta values
        % FOR EACH SAMPLE OF THE TRAINING SET 
        grad = zeros(m,1+size(X_train,2));
        for i=1:m
            x_i = [1, X_train(i,:)]; % Put a 1 (value of x0) at the beginning of the pattern
            % Expected output (i.e. result of the sigmoid function) for i-th pattern
            % ====================== YOUR CODE HERE ======================
            h_train(i) = fun_sigmoid(theta, x_i);
            % ============================================================
            grad(i,:) = (h_train(i)-Y_train(i))*x_i;
        end
        
        % *********************
        % STEP 2. Update the theta values. To do it, follow the update
        % equations that you studied in the theoretical session
        % ====================== YOUR CODE HERE ======================
        %A = [ones(m,1) ,X_train];
        theta = theta - alpha*(1/m)*(sum(grad));
        
        % ============================================================
        
        % *********************
        % STEP 3. Calculate the cost on this iteration and store it on
        % vector J.
        
        % ====================== YOUR CODE HERE ======================
        total_cost = 0;
        for i=1:m
            x_i = [1, X_train(i, :)]; % Put a 1 (value for x0) at the beginning of each pattern

            % Expected output (i.e. result of the sigmoid function) for i-th pattern
            % ====================== YOUR CODE HERE ======================
            h_train(i) = fun_sigmoid(theta, x_i);
            % ============================================================

            % Calculate the cost for the i-the pattern and add the cost of the last patterns
            % ====================== YOUR CODE HERE ======================
            total_cost = total_cost + fCalculateCostLogReg(Y_train(i), h_train(i));
            % ============================================================
        end
        % ============================================================
        
        J(1+num_iter) = (-1/m)*total_cost;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if VERBOSE
        % Dibuja el coste J en función del número de iteración
        figure;
        plot(0:num_iter, J, '-')
        title(['Cost function over the iterations with alfa=', num2str(alpha)]);
        xlabel('Number of iterations');
        ylabel('Cost J');
    end

end