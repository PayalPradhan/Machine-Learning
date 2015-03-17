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
	 %K=(sum(theta .*theta)*lambda)/(2*m);
	 K=0;
	 n=size(theta,1);
	 for i=2:n,
        K=K+theta(i,1)*theta(i,1);
     end
     K=K*lambda/(2*m);	 
	 J=J+K;
	 %calculating Gradient
	 n=size(theta,1);
	 grad(1,1)=sum((A-y).*X(:,1))/m;
	 for i=2:n
	   P=X(:,i);
	   grad(i,1)=(sum((A-y).*P)/m +lambda*theta(i,1)/m);
	  end





% =============================================================

end
