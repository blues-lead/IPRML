function cov_mat = get_covariance_matrix(class,data,label)
    idx = find(label == class);
    F = data(idx,:);
    cov_mat = cov(F);
end

