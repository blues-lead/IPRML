%A([1 3 5],:)... will get needed indecies
function [mu,sigma,p] = cifar_10_bayes_learn(F,labels)
    for i=0:9
        idx = find(labels == i);
        class_i = F(idx,:);
        mu(i+1,:) = mean(class_i);
        sigma(i+1,:) = std(class_i);
    end
    p = 1/10;
end