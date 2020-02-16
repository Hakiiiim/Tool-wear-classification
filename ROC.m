function [TPR,FPR] = ROC(Y0,Ytest)
    T = 0:0.001:1;
    [n, m] = size(T);
    
    TPR = zeros(1,m);
    FPR = zeros(1,m);
    
    for j=1:m
        Y1 = Ytest>=(T(j));
        
        Cfman = zeros(2,2);

        for i=1:81
           if (Y0(i) == 1) && (Y1(i) == 1) 
               Cfman(1,1) = Cfman(1,1)+1;
           elseif (Y0(i) == 0) && (Y1(i) == 1) 
               Cfman(2,1) = Cfman(2,1)+1;
           elseif (Y0(i) == 1) && (Y1(i) == 0) 
               Cfman(1,2) = Cfman(1,2)+1;
           elseif (Y0(i) == 0) && (Y1(i) == 0)
               Cfman(2,2) = Cfman(2,2)+1;   
           end
        end

        Cf = Cfman;
        
        tpr = Cf(1,1)/(Cf(1,1)+Cf(1,2));
        fpr = Cf(2,1)/(Cf(2,1)+Cf(2,2));
        
        TPR(j) = tpr;
        FPR(j) = fpr;
    end
end

