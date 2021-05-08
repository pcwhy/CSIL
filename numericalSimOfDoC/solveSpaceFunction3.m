clc
close all;
clear;
%rng default

numFPs = 4;
numDims = 2;
numLambdas = numFPs;

variableArray = {};
for i = 1:numFPs
    variableArray{end+1} = optimvar(strcat('x',string(i)),numDims);
end
for i = 1:numLambdas
    variableArray{end+1} = optimvar(strcat('lbd',string(i)),1);
end

prob = eqnproblem;

eqStruct = struct;
eqCount = 1;
expr = optimexpr;
for i = 1:(numLambdas)
    expr = sum(variableArray{i}.^2,'all') == 1;
    eqStruct = setfield(eqStruct, strcat('eq',string(eqCount)),expr); 
    eqCount = eqCount + 1;
end
expr = 0;
for i = 1:numFPs
    for j = 1:numDims
        for k = 1:numFPs
            if k ~= i
                expr = expr + variableArray{k}(j);
            else
                expr = expr + 2.*variableArray{i+numFPs}(1).*variableArray{k}(j);
            end
        end
        expr = expr == 0;
        eqStruct = setfield(eqStruct, strcat('eq',string(eqCount)),expr); 
        eqCount = eqCount + 1;
        expr = 0;
    end
end
prob.Equations = eqStruct;
x0 = struct;
for i = 1:numFPs
    rv = randn(numDims,1);
    rv = rv./vecnorm(rv);
    x0 = setfield(x0, strcat('x',string(i)),rv);
end
for i = 1:numFPs
    rv = 1;
    x0 = setfield(x0, strcat('lbd',string(i)),rv);
end
show(prob)
% prob.Objective.evaluate(x0)
[sol,fval,exitflag] = solve(prob,x0)
% prob.Objective.evaluate(sol)

FPs = [];
sol = struct2cell(sol);
for i = 1:numFPs
    FPs(end+1,:) = sol{numFPs+i}';
end
FPs
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
        for k = 1:size(weights,1)
            nxtWeight = weights(k,:);  
            magNxtWeight = sqrt(sum(nxtWeight.^2,2));
            similarMatrix(i,k)=sum((nxtWeight./magNxtWeight)...
                .*(curWeight./magCurWeight));
        end
    end
    imagesc(similarMatrix)
end

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
sumDeviationAngles = (sum(similarMatrix,'all') - sum(diag(similarMatrix)))./2





