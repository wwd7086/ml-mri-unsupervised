function missVoxel = superSparse(numPC, train,testProv,missIdx,provideIdx)
    %% learn a dictionary
    param.K=256;  % learns a dictionary with 100 elements
    param.lambda=0.15;
    param.numThreads=-1; % number of threads
    param.batchsize=400;
    param.verbose=false;
    param.iter=1000;  % let us see what happens after 1000 iterations.
    
    % D m*p  m->num of features
    % train' m*n
    tic
    %D = mexTrainDL_Memory(train',param); %m*p
    load sparse.mat
    t=toc;
    fprintf('time of computation for Dictionary Learning: %f\n',t);
    save('sparse.mat','D');

    
    %% find sparse coding using dictionary
    % parameter of the optimization procedure are chosen
    param.L=20; % not more than 20 non-zeros coefficients (default: min(size(D,1),size(D,2)))
    param.eps=0.01; % threshold for ther residual
    param.numThreads=-1; % number of processors/cores to use; the default choice is -1
                     % and uses all the cores of the machine
    
    mask = false(size(D,1),size(testProv,1));
    mask(provideIdx,:) = true;
    tic
    % alpha p*n
    fullProv = zeros(size(D,1),size(testProv,1));
    fullProv(provideIdx,:) = testProv';
    alpha=mexOMPMask(fullProv,D,mask,param);
    t = toc;
    fprintf('%f signals processed per second\n',size(testProv,1)/t);
    
    % reconstruct
    allVoxel = D*alpha; %m*p*p*n = m*n
    missVoxel = allVoxel(missIdx,:)';

end
