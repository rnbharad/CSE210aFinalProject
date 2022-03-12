
function C = getCmatrix(w)
    C=zeros(484,784);
    p=1;
    for k=0:21
        for j=0:21
            for i=1:7
                C(p,i+j+28*k)=w(i);
                C(p,i+j+28*k+28)=w(i+7);
                C(p,i+j+28*k+56)=w(i+14);
                C(p,i+j+28*k+84)=w(i+21);
                C(p,i+j+28*k+112)=w(i+28);
                C(p,i+j+28*k+140)=w(i+35);
                C(p,i+j+28*k+168)=w(i+42);
            end
            p=p+1;
        end
    end        
end