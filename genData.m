clear;
close all;

load Train.mat;
load Test.mat;

xall = [Xtest;Xtrain];
eventall = [eventsTest;eventsTrain];

X2train = xall(1:1201,:);
X2test = xall(1202:end,:);
Etrain = eventall(1:1201,:);
Etest = eventall(1202:end,:);

save('full.mat','X2train','X2test','Etrain','Etest');