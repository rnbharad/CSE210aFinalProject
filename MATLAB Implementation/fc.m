clear all
load('mnist2.mat');
%%%%%%%%%%%%%%%%%%%%%inputs%%%%%%%%%%%%%%%%%%%%%%
test = [Matrix1_Test, Matrix3_Test
        ones(1,50), zeros(1,50)
        zeros(1,50), ones(1,50)];
train = [Matrix1_Train, Matrix3_Train
        ones(1,17), zeros(1,17)
        zeros(1,17), ones(1,17)];
alpha = 0.02;
a=0;
c=1;
b1(1)=a+(c-a)*rand(1);
b1(2)=a+(c-a)*rand(1);
for p=1:2
    r = a+(c-a)*rand(1);
    for q=1:784
        w1(q,p)= r/100;
    end
end
b2(1)=a+(c-a)*rand(1);
b2(2)=a+(c-a)*rand(1);
for q=1:2
    for p=1:2
        w2(q,p)=a+(c-a)*rand(1);
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%train%%%%%%%%%%%%%%%%%%%%
for epoch = 1:100
    ind = randperm(34);
    for j = ind
        % Forward Pass
        % Hidden Layer
        for i = 1:2
            wx = train(1:784,j)'*w1(:,i);
            H = b1(i) + wx;
             Y(i) = sigmf(H,[1 0]);
            % Hyperbolic function.
            %Y(i) = tanh(H);
            % Relu function.
            % Y(i) = max(H,0);
        end
        % Output Layer
        sum = 0;
        for i = 1:2
            vx=0;
            for k = 1:2
                vx = vx + Y(k)* w2(i,k);
            end
            V = b2(i) + vx;
             out(i) = sigmf(V,[1 0]);
            %out(i) = tanh(V);
            % out(i) =  max(V,0);
            %calculate the sum-of-square error
            e(i) = (train(i+784,j)-out(i));
            sum = sum +(e(i).^2);
        end
        %back propagation
        % Output Layer (Bias)
        for i = 1:2
             delta3(i) = out(i)*(1-out(i))* e(i);
            %delta3(i) = (1-out(i)^2)* e(i);
            % delta3(i) = (out(i)>0)* e(i);
            b2(i)=b2(i)+alpha*delta3(i);
        end
        for i = 1:2
            delta21 = 0;
            for k = 1:2
                delta21 = delta21+Y(i)*(1-Y(i))*w2(k,i)*delta3(k);
                %delta21 = delta21+(1-Y(i)^2)*w2(k,i)*delta3(k);
                %delta21 = delta21+(Y(i)>0)*w2(k,i)*delta3(k);
                w2(k,i)=w2(k,i)+alpha*Y(i)*delta3(k);
            end
            % Hidden Layer
            b1(i)=b1(i)+alpha*delta21;
            w1(:,i) = w1(:,i) + alpha * train(1:784,j)*delta21;
        end
    end
    SSE(epoch)=sum/2;    
    %%%%%%%%%%%%%test%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    T=0;
    for j = 1:100
        % Forward Pass
        % Hidden Layer
        for i = 1:2
            wx = test(1:784,j)'*w1(:,i);
            H = b1(i) + wx;
             Y(i) = sigmf(H,[1 0]);
            % Hyperbolic function.
            %Y(i) = tanh(H);
            % Relu function.
            Y(i) = max(H,0);
        end
        % Output Layer
        sum1 = 0;
        for i = 1:2
            vx=0;
            for k = 1:2
                vx = vx + Y(k)* w2(i,k);
            end
            V = b2(i) + vx;
            out(i) = sigmf(V,[1 0]);
            %out(i) = tanh(V);
            % out(i) =  max(V,0);
            %calculate the sum-of-square error
            e(i) = (test(i+784,j)-out(i));
            sum1 = sum1 +(e(i).^2);
        end
        if(out(1) > out(2))
            if(test(785,j)==1)
                T=T+1;
            end
        else
            if(test(786,j)==1)
                T=T+1;
            end
        end
    end
    SSE1(epoch) = sum1/2;   
    Acc(epoch) = T;
end
figure(1), imshow(reshape(w1(:,1),[28,28]),[]);
figure(2), imshow(reshape(w1(:,2),[28,28]),[]);
figure, plot(SSE), ylabel('Training Error'), xlabel('epoch')
figure, plot(SSE1), ylabel('Testing Error'), xlabel('epoch')
figure, plot(Acc), ylabel('Accuracy'), xlabel('epoch')