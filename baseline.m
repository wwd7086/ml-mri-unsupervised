clear;
close all;

% load datas
load('full.mat');
load('provideIdx.mat');
load('missIdx.mat');
load('provideData_1000.mat');

% load libs
addpath spams-matlab/build;
addpath spams-matlab;
start_spams;
addpath KMeans++;
addpath libsvm-3.21/matlab/;

%% inputs
isTest = true;
if isTest
    doTest = true;
    doGen = false;
    vtrain = X2train;
    testProv = X2test(:,provideIdx);
    testMiss = X2test(:,missIdx);
else
    doTest = false;
    doGen = true;
    vtrain = [X2train;X2test];
    testProv = provideData;
end

%% learn PCA from full voxel images
%missVoxel = simplePCA(80,vtrain,testProv,missIdx,provideIdx);
%missVoxel = eventPCA(80,vtrain,testProv,missIdx,provideIdx);
%missVoxel = kmeanRecon(200,vtrain,testProv,missIdx,provideIdx);
%missVoxel = superSparse(200,vtrain,testProv,missIdx,provideIdx);
%missVoxel = nnSearch(200,vtrain,testProv,missIdx,provideIdx);
%missVoxel = simpleRegression(80,vtrain,testProv,missIdx,provideIdx);
%missVoxel = svRegression(80,vtrain,testProv,missIdx,provideIdx);
%missVoxel = svliRegression(80,vtrain,testProv,missIdx,provideIdx);
missVoxel = superNeural(80,vtrain,testProv,missIdx,provideIdx);

%% calculate erro
if doTest
    errs = missVoxel - testMiss;
    rmse = sqrt(mean(errs(:).^2))
end

%% generate csv
if doGen
    csvwrite('prediction.csv', missVoxel);
end
