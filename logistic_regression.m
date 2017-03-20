function []=logistic_regression(trFile ,degree ,tFile )
trainingData = dlmread(trFile);
testData = dlmread(tFile);
[nrow, ncollumn] = size(trainingData);
[nTrow, ~] = size(testData);
ncollumn = ncollumn - 1;
t = trainingData(:,end);
for k=1:nrow
    if t(k,1)~=1
        t(k,1)=0;
    end
end
tt = testData(:,end);
for k=1:nTrow
    if tt(k,1)~=1
        tt(k,1)=0;
    end
end


if degree == 1
    %BEGIN TRAINING
    w = zeros(ncollumn+1 ,1);
    phi = ones(nrow,1);
    phi = [phi, trainingData(:,1:end-1)];
    y = w'*phi';
    y = 1./(1+exp(-y));
    y = y';
    R = zeros(nrow, nrow);
    for i=1:nrow
        for j = 1:nrow
            if i==j
                R(i,j) = y(i,1)*(1-y(i,1));
            end
        end
    end
    crossEntropy = phi'*(y-t);
    wOld = w;
    w = w - pinv(phi'*R*phi)*crossEntropy;
    while sum(abs(wOld-w))>0.001
        
        y = w'*phi';
        y = 1./(1+exp(-y));
        y = y';
        R = zeros(nrow, nrow);
        for i=1:nrow
            for j = 1:nrow
                if i==j
                    R(i,j) = y(i,1)*(1-y(i,1));
                end
            end
        end
        crossEntropyOld = crossEntropy;
        crossEntropy = phi'*(y-t);
        wOld = w;
        w = w - pinv(phi'*R*phi)*crossEntropy;
        if (abs(crossEntropy-crossEntropyOld))<0.001
            break;
        end
    end
    [wsize,~] = size(w);
    for i=1:wsize
        fprintf('w%d=%.4f\n',i-1, w(i,1));
    end
    %END OF TRAINING
    %CLASSIFICATION
    phiT = ones(nTrow,1);
    phiT = [phiT, testData(:,1:end-1)];
    a = w'*phiT';
    yT = 1./(1+exp(-a));
    a = a';
    yT = yT';
    out = zeros(nTrow,1);
    acc = zeros(nTrow,1);
    for j=1:nTrow
        if a(j,1)>0
            if yT(j,1)>0.5
                out(j,1)=1;
                if out(j,1)==tt(j,1)
                    acc(j,1)=1;
                else 
                    acc(j,1)=0;
                end
            elseif yT(j,1)<0.5
                out(j,1)=0;
                yT(j,1)=1-yT(j,1);
                if out(j,1)==tt(j,1)
                    acc(j,1)=1;
                else
                    acc(j,1)=0;
                end
            else
                out(j,1)=1;
                acc(j,1)=0.5;
            end
        elseif a(j,1)<0 
            if (1-yT(j,1))>0.5
                out(j,1)=0;
                yT(j,1)=1-yT(j,1);
                if out(j,1)==tt(j,1)
                    acc(j,1)=1;
                else
                    acc(j,1)=0;
                end
            elseif (1-yT(j,1))<0.5
                out(j,1)=1;
                if out(j,1)==tt(j,1)
                    acc(j,1)=1;
                else
                    acc(j,1)=0;
                end
            else
                out(j,1)=1;
                acc(j,1)=0.5;
            end
        else
            out(j,1)=1;
            acc(j,1)=0.5;
        end
    end
    % END OF CLASSSIFICATION
    for i=1:nTrow
        fprintf('ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f\n',i-1, out(i,1), yT(i,1), tt(i,1), acc(i,1));
    end
    acc = sum(acc)/nTrow;
    fprintf('classification accuracy=%6.4f\n', acc);
    
elseif degree==2
    % TRAINING
    temp = zeros(nrow, 2*ncollumn);
    for i=1:nrow
        c=1;
        for j=1:ncollumn
            temp(i,c) = trainingData(i,j);
            temp(i,c+1) = trainingData(i,j)^2;
            c=c+2;
        end
    end
    w = zeros(2*ncollumn+1 ,1);
    phi = ones(nrow,1);
    phi = [phi, temp];
    y = w'*phi';
    y = 1./(1+exp(-y));
    y = y';
    R = zeros(nrow, nrow);
    for i=1:nrow
        for j = 1:nrow
            if i==j
                R(i,j) = y(i,1)*(1-y(i,1));
            end
        end
    end
    crossEntropy = phi'*(y-t);
    wOld = w;
    w = w - pinv(phi'*R*phi)*crossEntropy;
    
    while sum(abs(wOld-w))>0.001
        
        y = w'*phi';
        y = 1./(1+exp(-y));
        y = y';
        R = zeros(nrow, nrow);
        for i=1:nrow
            for j = 1:nrow
                if i==j
                    R(i,j) = y(i,1)*(1-y(i,1));
                end
            end
        end
        crossEntropyOld = crossEntropy;
        crossEntropy = phi'*(y-t);
        wOld = w;
        w = w - pinv(phi'*R*phi)*crossEntropy;
         if (abs(crossEntropy-crossEntropyOld))<0.001
            break;
         end
        
    end
    [wsize,~] = size(w);
    for i=1:wsize
        fprintf('w%d=%.4f\n',i-1, w(i,1));
    end
%   END OF TRAINING
% Classification
    tempT = zeros(nTrow, 2*ncollumn);
    for i=1:nTrow
        c=1;
        for j=1:ncollumn
            tempT(i,c) = testData(i,j);
            tempT(i,c+1) = testData(i,j)^2;
            c=c+2;
        end
    end
    phiT = ones(nTrow,1);
    phiT = [phiT, tempT];
    a = w'*phiT';
    yT = 1./(1+exp(-a));
    a = a';
    yT = yT';
    out = zeros(nTrow,1);
    acc = zeros(nTrow,1);
    for j=1:nTrow
        if a(j,1)>0
            if yT(j,1)>0.5
                out(j,1)=1;
                if out(j,1)==tt(j,1)
                    acc(j,1)=1;
                else 
                    acc(j,1)=0;
                end
            elseif yT(j,1)<0.5
                out(j,1)=0;
                yT(j,1)=1-yT(j,1);
                if out(j,1)==tt(j,1)
                    acc(j,1)=1;
                else
                    acc(j,1)=0;
                end
            else
                out(j,1)=1;
                acc(j,1)=0.5;
            end
        elseif a(j,1)<0 
            if (1-yT(j,1))>0.5
                out(j,1)=0;
                yT(j,1)=1-yT(j,1);
                if out(j,1)==tt(j,1)
                    acc(j,1)=1;
                else
                    acc(j,1)=0;
                end
            elseif (1-yT(j,1))<0.5
                out(j,1)=1;
                if out(j,1)==tt(j,1)
                    acc(j,1)=1;
                else
                    acc(j,1)=0;
                end
            else
                out(j,1)=1;
                acc(j,1)=0.5;
            end
        else
            out(j,1)=1;
            acc(j,1)=0.5;
        end
    end
    %end of classification
    for i=1:nTrow
        fprintf('ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f\n',i-1, out(i,1), yT(i,1), tt(i,1), acc(i,1));
    end
    acc = sum(acc)/nTrow;
    fprintf('classification accuracy=%6.4f\n', acc);
    
end
    

end