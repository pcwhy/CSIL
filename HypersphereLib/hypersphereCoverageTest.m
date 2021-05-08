clc
clear
close all;
cosAlpha = [0.3:0.1:0.9];
results = [];
results2 = [];
set(0,'DefaultTextFontName','Times','DefaultTextFontSize',18,...
   'DefaultAxesFontName','Times','DefaultAxesFontSize',18,...
   'DefaultLineLineWidth',1,'DefaultLineMarkerSize',7.75)
for i = 2:100
    hsa = HypersphereSurfArea(i,1);
    hsc = [];
    ratios = [];
    classNum = [];

    for j = 1:numel(cosAlpha)
        hsc(end+1) = hypersphereCapArea(i,1,1- cosAlpha(j) );
        ratios(end+1) = hsc(end)./hsa;
        classNum(end+1) = hsa./hsc(end);
    end
    results(end+1,:) = [i,ratios];
    results2(end+1,:) = [i,classNum];

end
figure
hold on;
legendStrings = {};
for i = 2:size(results,2)
    plot(results(:,1),results(:,i),'LineWidth',1.5);
    legendStrings{end+1} = string('$cos(\sigma)=$') + string(cosAlpha(i-1));
end
xlabel('Dimensions')
ylabel('Hypersphere coverage ratio / class');
legend(legendStrings,'Interpreter','latex');
% set(gca,'FontSize',12)

xlim([2,50])
grid on;
box off;

figure
legendStrings = {};
for i = 2:size(results,2)
    semilogy(results2(:,1),(results2(:,i)),'LineWidth',1.5);
    hold on;
    legendStrings{end+1} = string('$cos(\sigma)=$') + string(cosAlpha(i-1));
end
xlabel('Dimensions')
ylabel('Maximum number of classes');
legend(legendStrings,'Interpreter','latex');
% set(gca,'FontSize',12)

xlim([2,50])
grid on;
box off;
