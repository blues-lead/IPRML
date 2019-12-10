function c = cifar_10_bayes_classify(data,label,f,mu,p)
%data - training data
%label - training labels
    for i=0:9
        cov_mat = get_covariance_matrix(i,data,label);
        prob(i+1) = mvnpdf(f,mu(i+1,:),cov_mat)*p;
    end
    mx = max(prob);
    c = find(prob==mx);
end