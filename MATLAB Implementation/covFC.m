clear all
close all
%%%%%%%%%%%%%%%%%%%%%inputs%%%%%%%%%%%%%%%%%%%%%%
load('mnist.mat');
N = 100;
Xtest = zeros(784,10*N);
Ytest = zeros(10,10*N);
Xtrain = zeros(784,10*N);
Ytrain = zeros(10,10*N);
Err= zeros(100,1);
Err1= zeros(100,1);
Acc1= zeros(100,1);
Acc= zeros(100,1);

for i = 1:10
    for j = 1:N
        Xtest(:,N*(i-1)+j) = reshape(reshape(TestData(i,:,j),[28,28])', [784,1]);
        Xtrain(:,N*(i-1)+j) = reshape(reshape(TrainData(i,:,j),[28,28])', [784,1]);
        temp = zeros(10,1);
        temp(i,1) = 1;
        Ytest(:,N*(i-1)+j) = temp;
        Ytrain(:,N*(i-1)+j) = temp;
    end
end
%%%%%%%%%%%%%%%%initialization%%%%%%%%%%%%%%%%%%%
alpha = 0.02;
L=1;
a=0;
c=1;
for p=1:16
    r=a+(c-a)*rand(1);
    for q=1:49
        w1(p,q)= r*0.001;
    end
end
for p=1:16
    r=a+(c-a)*rand(1);
    b1(p,1)= r*0.001;
end
for o=1:10
    for u=1:7744
        w2(o,u)=(a+(c-a)*rand(1))*0.001;
    end
end
for o=1:10
    b2(o,1)=(a+(c-a)*rand(1))*0.001;
end
%%%%%%%%%%%%%%%%%%%%%%%%train%%%%%%%%%%%%%%%%%%%%
for epoch = 1:100
    epoch
    ind = randperm(N*10);
    for j = ind
        % Forward Pass
        % Hidden Layer
        Y1 = [];
        for z=1:16
              C = getCmatrix(w1(z,:));
              V1 = C * Xtrain(:,j);

              V1 = V1 + b1(z);
              % sigmoid(X)  Logistic or "sigmoid" function.
              % Y = 1.0 ./ (1 + exp(-x));
              Y1 = [Y1; (1.0 ./ (1 + exp(-L*V1)))];
              % Hyperbolic function.
              % Y1 = [Y1; (tanh(V1))];
              % Relu function.
              % Y1 = [Y1; max(V1,0)];
        end
        % Output Layer
        V2 = w2 * Y1;
        V2 = V2 + b2;
        Y2 = 1.0 ./ (1.0 + exp(-L*V2));
        % Y2 = tanh(V2);
        % Y2 =  max(V2,0);
        e = (Ytrain(:,j) - Y2);
        E = sum(e.^2);
        %back propagation
        % Output Layer (Bias)
        delta = L * e .* (Y2.*(1.0-Y2));
        % delta = e .* (1.0-(Y2.^2));
        % delta = e .* (Y2>0);
        delta21 = delta' * w2;
        w2 = w2 + alpha * delta * Y1';
        b2 = b2 + alpha * delta;
        % Hidden Layer
        X=reshape(Xtrain(:,j), [28, 28])';                 
         xc=[];
         for i=1:7
             for x=1:7
                 xc=[xc, reshape((X(i:(21+i),x:(21+x)))',[22^2,1])];
             end
         end
        dY=delta21'.*Y1.*(1-Y1);
        % dY=delta21'.*(1-(Y1.^2));
        % dY=delta21'.*(Y1>0);
        gg = reshape(dY,[484,16])';
        w1=w1+alpha*gg*xc;
        b1=b1+alpha*sum(gg,2);
        Err(epoch)=Err(epoch) + E/2; 
    end
    %%%%%%%%%%%%%test%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    T=0;
    for j = 1:N*10
        % Forward Pass
        % Hidden Layer
        Y1 = [];
        for z=1:16
              C = getCmatrix(w1(z,:));
              V1 = C * Xtest(:,j);
              V1 = V1 + b1(z);
              % sigmoid(X)  Logistic or "sigmoid" function.
              % Y = 1.0 ./ (1 + exp(-x));
              Y1 = [Y1; (1.0 ./ (1 + exp(-L*V1)))];
              % Hyperbolic function.
              % Y1 = [Y1; (tanh(V1))];
              % Relu function.
              % Y1 = [Y1; max(V1,0)];
        end
        % Output Layer
        V2 = w2 * Y1;
        V2 = V2 + b2;
        Y2 = 1.0 ./ (1.0 + exp(-L*V2));
        %Y2 = tanh(V2);
        %Y2 = max(V2,0);
        [Ro,Io] = max(Y2);
        [Rd,Id] = max(Ytest(:,j));
        if(Io == Id)
            T=T+1;
        end
        e = (Ytest(:,j) - Y2);
        E1 = sum(e.^2);
        Err1(epoch)=Err1(epoch) + E1/2;
        Acc1(epoch)=Acc1(epoch) + T;
    end
     Acc(epoch)=T;
end
%%%%%%%%%%%%%%%%%%%%%outputs%%%%%%%%%%%%%%%%%%%%%
for i = 1:16
    subplot(4,4,i);
    imshow(reshape(w1(i,:),[7,7])',[]);
end
figure, plot(Err), ylabel('Training Error'), xlabel('epoch');
figure, plot(Err1), ylabel('Testing Error'), xlabel('epoch');
figure, plot(Acc1), ylabel('Accuracy'), xlabel('epoch');
figure, plot(Acc), ylabel('Accuracy'), xlabel('epoch');