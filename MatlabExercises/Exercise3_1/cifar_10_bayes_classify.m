function c = cifar_10_bayes_classify(f,mu,sigma,p)
    %f = cifar_10_features(f);
    for i=0:9
        j = i + 1;
        prob_r = normcdf(f(1),mu(j,1),sigma(j,1));
        prob_g = normcdf(f(2),mu(j,2),sigma(j,2));
        prob_b = normcdf(f(3),mu(j,3),sigma(j,3));
        prob(j) = prob_r*prob_g*prob_b*p;
    end
    mx = max(prob);
    c = find(prob==mx);
    %name=lab_names(c);
end