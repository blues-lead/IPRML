function f = cifar_10_features(x)
    sz = size(x);
    for i=1:sz[1];
        f(i,:)= [mean(x(i,1:1024)) mean(x(i,1025:2048)) mean(x(i,2049:3072))];
    end
end