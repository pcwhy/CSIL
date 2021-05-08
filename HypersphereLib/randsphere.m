function X = randsphere(m,n,r)
 % Roger Stafford (2020). Random Points in an n-Dimensional Hypersphere (https://www.mathworks.com/matlabcentral/fileexchange/9443-random-points-in-an-n-dimensional-hypersphere), MATLAB Central File Exchange. Retrieved November 9, 2020.
 
% This function returns an m by n array, X, in which 
% each of the m rows has the n Cartesian coordinates 
% of a random point uniformly-distributed over the 
% interior of an n-dimensional hypersphere with 
% radius r and center at the origin.  The function 
% 'randn' is initially used to generate m sets of n 
% random variables with independent multivariate 
% normal distribution, with mean 0 and variance 1.
% Then the incomplete gamma function, 'gammainc', 
% is used to map these points radially to fit in the 
% hypersphere of finite radius r with a uniform % spatial distribution.
% Roger Stafford - 12/23/05
 
X = randn(m,n);
s2 = sum(X.^2,2);
% X = X.*repmat(r*(gammainc(s2/2,n/2).^(1/n))./sqrt(s2),1,n);

X = X.*repmat(r*(rand(m,1).^(1/n))./sqrt(s2),1,n);

