function Y = HyperSphere(X,varargin)
% Gianluca Dorini (2020). hypersphere (https://www.mathworks.com/matlabcentral/fileexchange/5397-hypersphere), MATLAB Central File Exchange. Retrieved November 9, 2020.
% n-dimensional hypersphere
% 
% syntax: Y = HyperSphere(X,r)
%         Y = HyperSphere(X)
% 
% INPUT ARGUMENTS
%     
%         -X      (n - 1) x (N) every row of this matrix is a tuple
%                 of n-1 parametric coordinates. If the number of points is 
%                 N, the number of rows is N
%                 
%         -r      the radius, default value: r = 1
%         
% OUTPUT ARGUMENTS
% 
%         Y       (n) x (N) every row of this matrix is a tuple 
%                 of n  catesian coordinates of the n-dimensional hypersphere
% 
% this function calculates the cartesian coordinates of an n-dimensional hypersphere
% given the n-1 dimensions vector 'X' of parametric coordinates and the
% radius 'r'
% 
% 
% The n-hypersphere (often simply called the n-sphere) is a generalization 
% of the circle (2-sphere) and usual sphere (the 3-sphere) to dimensions n > 3. 
% The n-sphere is therefore defined as the set of n-tuples of points 
% (x1,x2,x3, ...,xn ) such that 
%     
% x1^2 + x2^2 + x3^2 + ... + xn^2 = r^2
% 
% where r is the radius of the hypersphere. 
% 
% 
% EXAMPLES
% 
%
% % linspace points on a 2-sphere line:
%
% N = 20;
% X = linspace(0,2*pi, N);
% r = 2;
% Y = HyperSphere(X,r);
% plot(Y(1,:),Y(2,:),'r.');
% 
% 
% % random points on 3-sphere surface:
%
% N = 1000;
% X = 2*pi*rand(2,N);
% r = .5;
% Y = HyperSphere([X],r);
% plot3(Y(1,:),Y(2,:),Y(3,:),'r.');
if nargin == 2
    r = varargin{1};
else
    r = 1;
end
% % % % % N = size(X,1) + 1;                  
% % % % % n = size(X,2);                      
% % % % % Y = ones(N,n);
% % % % % for i = 1:N - 1
% % % % %     y = [repmat(cos(X(i,:)),i,1)];
% % % % %     y = [y ; sin(X(i,:))];
% % % % %     y = [y ; ones(N - i - 1,n)];
% % % % %     
% % % % %     Y = Y.*y;
% % % % % end
% % % % % Y = Y*r;

% 
m = size(X,1);
% number of input vectors
n = size(X,2);

% reverse the order of X
X = X(m:-1:1,:);

% sine terms
S = [sin(X); ones(1,n)];
% cosine terms
C = [ones(1,n); cumprod(cos(X),1)];
% calculate output
Y = r * C .* S;

% reverse the order of Y
Y = Y(m+1:-1:1,:);