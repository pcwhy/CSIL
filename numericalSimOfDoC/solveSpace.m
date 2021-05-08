clc
close all;
clear;
% rng default

x1 = optimvar('x1',3);
x2 = optimvar('x2',3);
x3 = optimvar('x3',3);

prob = optimproblem;
% prob.Constraints.cons1 = x1(1)^2 + x1(2)^2 + x1(3).^2 == 1;
% prob.Constraints.cons2 = x2(1)^2 + x2(2)^2 + x2(3).^2 == 1;
% prob.Constraints.cons3 = x3(1)^2 + x3(2)^2 + x3(3).^2 == 1;
prob.Constraints.cons1 = sum(x1.^2,'all') == 1;
prob.Constraints.cons2 = sum(x2.^2,'all') == 1;
prob.Constraints.cons3 = sum(x3.^2,'all') == 1;

prob.Objective = sum(x1.*x2 + x1.*x3 + x2.*x3,'all');
% prob.Objective = sum(x1.*x2.*x3,'all');

x01 = randn(3,1);
x02 = randn(3,1);
x03 = randn(3,1);
x0.x1=x01./vecnorm(x01);
x0.x2=x02./vecnorm(x02);
x0.x3=x03./vecnorm(x03);
sum(x0.x1.*x0.x2+x0.x1.*x0.x3+x0.x2.*x0.x3,'all')
[sol,fval,exitflag] = solve(prob,x0);
sum(sol.x1.*sol.x2+sol.x1.*sol.x3+sol.x2.*sol.x3,'all')







