function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Part 1
% Concatenate intercept factor
a1 = [ones(size(X,1),1) X];  % 5000x(400+1)

% Get hidden layer output and concatenate intercept factor
z2 = a1*Theta1';  % 5000x401 * 401x25
a2 = [ones(size(a1,1),1) sigmoid(z2)];  % 5000x(25+1)

% Get prediction based on index where size(pred)=[5000 1]
z3 = a2*Theta2';  % 5000x26 * 26x10
a3 = sigmoid(z3);  % 5000x10

% One hot coding using auto broadcasting
y_one_hot = y==1:max(y);  % 5000x10

% Cost Function w/o regularization (unfinished)
J = (1/m)*sum(sum(-y_one_hot.*log(a3) - (1-y_one_hot).*log(1-a3)))...
    + (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));

% Part 2
% a1, z2, a2, z3, a3 are calculated as above

% Get error in output layer
e3 = a3-y_one_hot;  % 5000x10
e2 = e3*Theta2(:, 2:end).*sigmoidGradient(z2);  % 5000x10 * 10x(26-1) .* 5000x25
%       Note: error is corresponding to nodes, which is the contribution of
%             cost from each node. Therefore, Theta2 need to be removed the
%             1st column, which is the weight of bias (bias is additional added
%             all-one column in a2, not a node in NN)

% Get delta (size is same as Theta so that all weights are updated)
delta2 = e3'*a2;  % 10x5000 * 5000x26
delta1 = e2'*a1;  % 25x5000 * 5000x401

% Get gradient with regularization, excluding regularization for bias term in theta
Theta2_grad = (1/m)*delta2 + (lambda/m)*[zeros(size(Theta2,1),1) Theta2(:,2:end)];
Theta1_grad = (1/m)*delta1 + (lambda/m)*[zeros(size(Theta1,1),1) Theta1(:,2:end)];

% Unroll grad
grad = [Theta1_grad(:); Theta2_grad(:)];





% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
