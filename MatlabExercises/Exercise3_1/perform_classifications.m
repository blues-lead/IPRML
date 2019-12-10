cifar_10_read_data
sz = size(te_data);
A = cifar_10_features(te_data);
B = cifar_10_features(tr_data);
[mu,sigma,p] = cifar_10_bayes_learn(B,tr_labels);
for i=1:sz(1)
    predict(i)=cifar_10_bayes_classify(A(i,:),mu,sigma,p);
end
res = cifar_10_evaluate(predict',te_labels)