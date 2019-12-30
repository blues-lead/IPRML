%cifar_10_read_data
sz = size(te_data);
A = cifar_10_features(te_data);
B = cifar_10_features(tr_data);
[mu,p] = cifar_10_bayes_learn(B,tr_labels);
for i=1:sz(1)
    %tst = cifar_10_bayes_classify(B,tr_labels,A(i,:),mu,p)
    %x = [i,'iteration'];
    %disp(x)
    predict(i)=cifar_10_bayes_classify(B,tr_labels,A(i,:),mu,p);
end
res = cifar_10_evaluate(predict',te_labels)