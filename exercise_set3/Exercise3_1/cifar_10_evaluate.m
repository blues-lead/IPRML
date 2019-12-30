function accuracy = cifar_10_evaluate(pred, gt)
    
    accuracy = numel(find(pred(:)==gt(:)))/length(gt);
    
end