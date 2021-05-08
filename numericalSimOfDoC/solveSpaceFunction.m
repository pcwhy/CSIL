clc
close all;
clear;
% rng default

numFPs = 3;
numDims = 2;

variableArray = {};
for i = 1:numFPs
    variableArray{end+1} = optimvar(strcat('x',string(i)),numDims);
end

prob = optimproblem;

expr = optimexpr(1,1);
count = 0;
for i = 1:numFPs
    for j = 1:i-1
        expr = expr + variableArray{i}.*variableArray{j};
        count = count + 1;
    end
end
expr = sum(expr);
show(expr);
% prob.Objective = sum(x1.*x2 + x1.*x3 + x2.*x3,'all');
prob.Objective = expr;

consStruct = struct;
consCount = 1;
expr = optimexpr;
for i = 1:(numFPs)
    expr = sum(variableArray{i}.^2,'all') == 1;
    consStruct = setfield(consStruct, strcat('cons',string(consCount)),expr); 
    consCount = consCount + 1;
end
% for i = 1:numFPs
%     for j = 1:i-1
%         expr = variableArray{i}.*variableArray{j} <= 0.5;
%         consStruct = setfield(consStruct, strcat('cons',string(consCount)),expr); 
%         consCount = consCount + 1;
%     end
% end

prob.Constraints = consStruct;


x0 = struct;
for i = 1:numFPs
    rv = randn(numDims,1);
    rv = rv./vecnorm(rv);
    x0 = setfield(x0, strcat('x',string(i)),rv);
end
% x01 = randn(3,1);
% x02 = randn(3,1);
% x03 = randn(3,1);
% x0.x1=x01./vecnorm(x01);
% x0.x2=x02./vecnorm(x02);
% x0.x3=x03./vecnorm(x03);
prob.Objective.evaluate(x0)
[sol,fval,exitflag,output,lambda] = solve(prob,x0)
prob.Objective.evaluate(sol)

FPs = [];
sol = struct2cell(sol);
lambda.Constraints
for i = 1:numFPs
    FPs(end+1,:) = sol{i}';
end
% FPs
if numDims == 2
    figure;
    plot(0,0,'o','LineWidth',2);
    hold on;
    plot(FPs(:,1),FPs(:,2),'*','LineWidth',2)
else
    weights = FPs;
    similarMatrix = zeros(size(weights,1), size(weights,1));
    for i = 1:size(weights,1)
        curWeight = weights(i,:);  
        magCurWeight = sqrt(sum(curWeight.^2,2));
        for j = 1:size(weights,1)
            nxtWeight = weights(j,:);  
            magNxtWeight = sqrt(sum(nxtWeight.^2,2));
            similarMatrix(i,j)=sum((nxtWeight./magNxtWeight)...
                .*(curWeight./magCurWeight));
        end
    end
    imagesc(similarMatrix)
end





