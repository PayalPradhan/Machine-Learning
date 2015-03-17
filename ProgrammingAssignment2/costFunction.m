function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
     Z=zeros(m,1);
	 for i=1:m,
	   Z(i,1)=X(i,:)*theta;
	 end
	 A=sigmoid(Z);
	 B=log(A);
	 C=log(1-A);
	 D=-y .*B;
	 E=(1-y) .*C;
	 F=D-E;
	 G=sum(F);
	 J=G/m;
	 
	 
	 % Me:Now computing the gradient
	 P=X(:,1);
	 Q=X(:,2);
	 R=X(:,3);
	 S=A-y;
	 r1=S .*P; % calculating (sigmoid-y)*x(i)j
	 r2=S .*Q;
	 r3=S .*R;
	 
	 s1=sum(r1)/m;
	 s2=sum(r2)/m;
	 s3=sum(r3)/m;
	 
	 grad=[s1;s2;s3];
%
% Note: grad should have the same dimensions as theta
%








% =============================================================

end
