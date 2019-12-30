function f = cifar_10_features(x,N)
    vec_len = size(x);
    count =(vec_len(2)/(N*N))/3; %16x16; pic_size = 4, 4x4; pic_size = 16
    pic_size = 1024/count;
    for i=1:vec_len(1)
        f(i,:) = mean(reshape(x(i,:),pic_size,count*3));
    end
end