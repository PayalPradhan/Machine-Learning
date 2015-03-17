function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
s=size(z);
r=s(1,1); % Number of rows
c=s(1,2); % Number of column
for i=1:r,
 for j=1:c,
    f=exp(-1*z(i,j));
	g(i,j)=1/(1+f);
 end
end





% =============================================================

end
