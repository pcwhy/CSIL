addpath('../../')
addpath('../')
clear;
close all;
tic;load('./adsb-107loaded.mat');toc;
load('./adsb-107CSNet20.mat');

X2 = X;
Y2 = Y;
cX2 = cX;
cY2 = cY;

uX = X(:,:,:,logical(double(Y>20) .* double(Y<=40)));
uY = Y(logical(double(Y>20) .* double(Y<=40)));
cuX = cX(:,:,:,logical(double(cY>20) .* double(cY<=40)));
cuY = cY(logical(double(cY>20) .* double(cY<=40)));

X = X(:,:,:,Y <=20);
cX = cX(:,:,:,cY <= 20);
Y = Y(Y <= 20);
cY = cY(cY <= 20);

skipDAGNet = 0;
% CSIL
% CSILPast5000
% CSILLockOldFPs
% CSILLockOldFPsPast5000
% CSILLockOldFPsChessBoard
CSILLockOldFPsChessBoardPast5000

X = X2;
Y = Y2;
cX = cX2;
cY = cY2;

%%
uX = X(:,:,:,logical(double(Y>40) .* double(Y<=60)));
uY = Y(logical(double(Y>40) .* double(Y<=60)));
cuX = cX(:,:,:,logical(double(cY>40) .* double(cY<=60)));
cuY = cY(logical(double(cY>40) .* double(cY<=60)));

X = X(:,:,:,Y <=40);
cX = cX(:,:,:,cY <= 40);
Y = Y(Y <= 40);
cY = cY(cY <= 40);

skipDAGNet = 1;
% CSIL
% CSILPast5000
% CSILLockOldFPs
% CSILLockOldFPsPast5000
% CSILLockOldFPsChessBoard
CSILLockOldFPsChessBoardPast5000

X = X2;
Y = Y2;
cX = cX2;
cY = cY2;

%%
uX = X(:,:,:,logical(double(Y>60) .* double(Y<=80)));
uY = Y(logical(double(Y>60) .* double(Y<=80)));
cuX = cX(:,:,:,logical(double(cY>60) .* double(cY<=80)));
cuY = cY(logical(double(cY>60) .* double(cY<=80)));

X = X(:,:,:,Y <=60);
cX = cX(:,:,:,cY <= 60);
Y = Y(Y <= 60);
cY = cY(cY <= 60);

skipDAGNet = 1;
% CSIL
% CSILPast5000
% CSILLockOldFPs
% CSILLockOldFPsPast5000
% CSILLockOldFPsChessBoard
CSILLockOldFPsChessBoardPast5000

X = X2;
Y = Y2;
cX = cX2;
cY = cY2;
%%
uX = X(:,:,:,logical(double(Y>80) .* double(Y<=100)));
uY = Y(logical(double(Y>80) .* double(Y<=100)));
cuX = cX(:,:,:,logical(double(cY>80) .* double(cY<=100)));
cuY = cY(logical(double(cY>80) .* double(cY<=100)));

X = X(:,:,:,Y <=80);
cX = cX(:,:,:,cY <= 80);
Y = Y(Y <= 80);
cY = cY(cY <= 80);

skipDAGNet = 1;
% CSIL
% CSILPast5000
% CSILLockOldFPs
% CSILLockOldFPsPast5000
% CSILLockOldFPsChessBoard
CSILLockOldFPsChessBoardPast5000

X = X2;
Y = Y2;
cX = cX2;
cY = cY2;
