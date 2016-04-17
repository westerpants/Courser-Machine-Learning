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

X=[ones(m,1) X];
a2=sigmoid(X*Theta1');

% X: 5000 x 401 , Theta1' : 401 x 25
% a2: 5000 x 25

a2=[ones(size(a2,1),1) a2];

% a2: 5000 x 26, Theta2' : 26x10
% a3: 5000 x 10
a3=a2*Theta2';

h=sigmoid(a3);
%h: 5000 x 10
%y : 5000 x 1

y_matrix=zeros(m, num_labels);
for i=1:m
  y_matrix(i,y(i))=1;
end

%y_matrix: 5000*10

for i=1:m
  J=J+sum(-y_matrix(i,:).*log(h(i,:)))-sum((1-y_matrix(i,:)).*log(1-h(i,:)));
end

J=J/m;

J = J + lambda * (sum(sum(Theta1(:, 2:end).^2)) + sum(sum(Theta2(:, 2:end).^2)))/ (2*m);




for t=1:m
  %Step 1	
  a1=X(t,:); %a1: 1x401, Theta1': 401x25
  z2=a1*Theta1'; %z2: 1x25
  a2=sigmoid(z2); %a2: 1x25
  a2=[1 a2]; %a2: 1x26
  z3=a2*Theta2'; %Theta2': 26x10
  a3=sigmoid(z3); %a3: 1x10

  %Step 2
  delta3=a3-y_matrix(t,:);
  %delta3: 1x10

  %Step 3
  delta2= delta3*Theta2 .* sigmoidGradient([1 z2]);
  %delta3: 1x10, Theta2: 10x26
  %delta3*Theta2: 1x26 
  %delta2: 1x26 
  
  delta2=delta2(2:end);
  %delta2: 1x25

  %Step 4

  Theta2_grad = Theta2_grad + delta3'*a2;
  %Theta2_grad: 10x26
  Theta1_grad = Theta1_grad + delta2'*a1;
  %Theta1_grad: 25x401
end


Theta1(:,1)=zeros(size(Theta1,1),1);
Theta2(:,1)=zeros(size(Theta2,1),1);

Theta2_grad=Theta2_grad./m + Theta2.*lambda/m;
Theta1_grad=Theta1_grad./m + Theta1.*lambda/m;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
