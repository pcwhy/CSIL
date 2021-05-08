function [dataTensor,cvDataTensor,label,cvLabel] = makeDataTensor(selectedBasebandData,selectedRawCompData)


selectedNoiseVector = zeros(size(selectedBasebandData));
selectedNoiseVector(:,1) = selectedBasebandData(:,1);
medoidVector = zeros(size(selectedBasebandData));
for i = 1:size(selectedBasebandData,1)
    [selectedNoiseVector(i,2:size(selectedBasebandData,2)),medoids]...
        = extractNoise(selectedBasebandData(i,2:size(selectedBasebandData,2)));
    medoidVector(i,1:1024) = medoids;
% plot(selectedNoiseVector(1,2:size(selectedBasebandData,2)),'.','LineWidth',1)
% hold on;
% plot(selectedBasebandData(i,2:size(selectedBasebandData,2)),'--','LineWidth',1)
% plot(medoids,'LineWidth',1);
% xlim([1,300])
% ylim([-0.1,0.3])
% set(gca,'FontSize',12)
% %set(gcf,'position',[405   330   528   279]);
% set(gcf,'position',[405   417   441   192]);
% legend('Pesudo noise','Baseband signals','Medoids')
% box off
end
cvNoise = selectedNoiseVector(ceil(0.7*size(selectedNoiseVector,1)):size(selectedNoiseVector,1),:);
selectedNoiseVector = selectedNoiseVector(1:ceil(0.7*size(selectedNoiseVector,1))-1,:);


% noiseSample = cvNoise(:,2:end);
% corrvals = [];
% noiseCorrvals = [];
% for i = 1:size(noiseSample,1)
% 	r = corrcoef(noiseSample(i,:),selectedBasebandData(i,2:end));
%     nr = corrcoef(randn(size(selectedBasebandData(i,2:end))),selectedBasebandData(i,2:end));
% 	corrvals(end+1)=r(2,1);
%     noiseCorrvals(end+1) = nr(2,1);
% end
% histogram(corrvals,'Normalization','probability')
% hold on;
% histogram(noiseCorrvals,'Normalization','probability')
% set(gca,'FontSize',12)
% set(gcf,'position',[405   417   441   147]);
% box off
% xlabel('Correlation coefficients')
% ylabel('Probability')
% legend('Pesudo noise','Gaussian noise ~ N(0,1)');


selectedFFTVector = zeros(size(selectedBasebandData));
selectedFFTVector(:,1) = selectedBasebandData(:,1);
for i = 1:size(selectedBasebandData,1)
%   selectedFFTVector(i,2:end) = fftshift(fft(selectedBasebandData(i,2:end)));
    actualFFT = fftshift(fft(selectedRawCompData(i,:),1024));
    rationaleFFT = fftshift(fft(medoidVector(i,:),1024));
    %selectedFFTVector(i,2:end) = fftshift(fft(selectedRawCompData(i,:),1024));
    selectedFFTVector(i,2:end) = actualFFT-rationaleFFT;
end
cvFFT = selectedFFTVector(ceil(0.7*size(selectedFFTVector,1)):size(selectedFFTVector,1),:);
selectedFFTVector = selectedFFTVector(1:ceil(0.7*size(selectedFFTVector,1))-1,:);
selectedFFTmag = abs(selectedFFTVector);
selectedFFTang = angle(selectedFFTVector);
%selectedFFTang = unwrap(atan2(real(selectedFFTVector),imag(selectedFFTVector)));

cvFFTmag = abs(cvFFT);
cvFFTang = angle(cvFFT);
%cvFFTang = unwrap(atan2(real(cvFFT),imag(cvFFT)));


featureDims = 3;

trainDataTensor = zeros(size(selectedNoiseVector,1)*featureDims,...
    size(selectedNoiseVector,2)-1);

for i = 1:size(selectedNoiseVector,1)
    cursor = i*featureDims-(featureDims-1);
    trainDataTensor(cursor,:) = selectedNoiseVector(i,2:end);
    trainDataTensor(cursor+1,:) = real(selectedFFTVector(i,2:end));
    trainDataTensor(cursor+2,:) = imag(selectedFFTVector(i,2:end));    
end

cvDataTensor = zeros(size(cvNoise,1)*featureDims, size(cvNoise,2)-1);
for i = 1:size(cvNoise,1)
    cursor = i*featureDims-(featureDims-1);    
    cvDataTensor(cursor,:) = cvNoise(i,2:end);    
    cvDataTensor(cursor+1,:) = real(cvFFT(i,2:end));
    cvDataTensor(cursor+2,:) = imag(cvFFT(i,2:end));
end


label = selectedNoiseVector(:,1);
cvLabel = cvNoise(:,1);

dataTensor = reshape(trainDataTensor',[sqrt(1024), sqrt(1024),...
    featureDims, size(selectedNoiseVector,1)]);
cvDataTensor = reshape(cvDataTensor',[sqrt(1024), sqrt(1024),...
    featureDims, size(cvLabel,1)]);

% One way to restore the origin signal is:
% sig = X2(:,:,:,1);
% sig2 = reshape(sig,[1,1024]);




