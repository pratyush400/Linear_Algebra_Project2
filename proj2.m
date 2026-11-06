    
    %--- Question 1
    
    [imgs, labels] = readMNIST('train-images-idx3-ubyte','train-labels-idx1-ubyte',5000,0);

    %--- Question 2

    [matrix_count, accepted] = extractor(imgs);

    %--- Question 3

    [y_train, A_train] = design_matrix_builder(imgs, labels, accepted);
    [theta_train, T_train] = least_squares(y_train, A_train, accepted, size(imgs,1), size(imgs,2));
    title('\theta weights, 5000 training images');

    %--- Question 4

    [imgs_test, labels_test] = readMNIST('t10k-images-idx3-ubyte','t10k-labels-idx1-ubyte',5000,0);
    [y_test, A_test] = design_matrix_builder(imgs_test, labels_test, accepted);      
    [y_pred, test_error] = compute_error(A_test, theta_train, y_test);
    fprintf('5000 sample test error: %.2f%%\n', 100 * test_error);
    [false_pos_rate, false_neg_rate, pos_y, neg_y, y_minus_sample, y_plus_sample] = compute_fpnr(y_test, y_pred);
    fprintf('theta_0: %.4f\n', theta_train(1));
    fprintf('falsepos: %.4f\n falseneg: %.4f\n',false_pos_rate, false_neg_rate )
    fprintf('A: %d x %d, y: %d x 1\n', size(A_train,1), size(A_train,2), size(y_train,1));
    fprintf('M0 = %d, M0+1 = %d (must be < 400 for the 400-image run)\n', size(accepted,1), size(accepted,1)+1);

    %--- Question 5

    [imgs400, labels400] = readMNIST('train-images-idx3-ubyte','train-labels-idx1-ubyte',400,0);
    [y_train400, A_train400] = design_matrix_builder(imgs400, labels400, accepted);
    [theta_train400, T_train400] = least_squares(y_train400, A_train400, accepted, size(imgs400,1), size(imgs400,2));
    title('\theta weights, 400 training images');
    [y_pred400, test_error400] = compute_error(A_test, theta_train400, y_test);
    [fpr400, fnr400] = compute_fpnr(y_test, y_pred400);
    
    fprintf('5000 train: err %.2f%%, FPR %.2f%%, FNR %.2f%%\n', 100*test_error, 100*false_pos_rate, 100*false_neg_rate);
    fprintf(' 400 train: err %.2f%%, FPR %.2f%%, FNR %.2f%%\n', 100*test_error400, 100*fpr400, 100*fnr400);
    
    %--- Question 6
    
    M1 = 100;
    [F_train_100, F_test_100, R] = random_feature_map(A_train, A_test, M1);

    %--- Question 7

    rand_list = [10, 20, 50, 100, 1000, 1500, 2000, 3000, 4000, 4500, 4600, 4700, 4800, 4900];
    test_errs = zeros(size(rand_list));
    
    for k = 1:length(rand_list)
        M1 = rand_list(k);
        [F_train, F_test, R] = random_feature_map(A_train, A_test, M1);
        A_new_train = [ones(size(F_train,1),1), F_train];
        A_new_test  = [ones(size(F_test,1),1),  F_test];
        theta_new = A_new_train \ y_train;
        [y_pred_new, test_errs(k)] = compute_error(A_new_test, theta_new, y_test);
        [fprs(k), fnrs(k)] = compute_fpnr(y_test, y_pred_new);
        fprintf('M1 = %4d: err %.2f%%, FPR %.2f%%, FNR %.2f%%\n', M1, 100*test_errs(k), 100*fprs(k), 100*fnrs(k));
    end

    figure; semilogx(rand_list, 100*test_errs, '-o');  title('Test error vs. number of random features'); grid on;
    xlabel('M_1'); ylabel('Test error rate (%)');

    %---functions---%

    function [matrix_count, accepted] = extractor(imgs) %counts number of non-zero intensities per pixel, generates a matrix with all pixels with more than 600 non-zero entries
    [num_rows, num_cols, total_images] = size(imgs);
    matrix_count = zeros(num_rows, num_cols); %number of non-zero entries per pixel
    for n = 1:total_images
        for i = 1:num_rows
            for j = 1:num_cols
                if imgs(i,j,n) > 0
                    matrix_count(i,j) = matrix_count(i,j) + 1;
                end
            end
        end
    end
    accepted = []; %list of all pixels with more than 600 non-zero entries
    for i = 1:num_rows
        for j = 1:num_cols
            if matrix_count(i,j) >= 600
                accepted(end+1,:) = [i,j,matrix_count(i,j)];
            end
        end
    end
    end
    
    function [y, A] = design_matrix_builder(imgs, labels, accepted) %builds the A matrix and y vector using the accepted pixels and label values
    total_images = size(imgs, 3);
    y = ones(length(labels),1);
    y(labels > 0) = -1; %all zeros are 1, non-zeros are -1
    M = size(accepted, 1);
    A = zeros(total_images, M+1);
    A(:,1) = 1;
    for n = 1:total_images %runs through all images and lists intensity of 'accepted' pixels
        for k = 1:M
            A(n,k+1) = imgs(accepted(k,1), accepted(k,2), n);
        end
    end
    end
    
    function [theta, T] = least_squares(y, A, accepted, num_rows, num_cols) %solves the least squares problem
    theta = A \ y; %theta
    M = size(accepted, 1);
    T = zeros(num_rows, num_cols); %visual representation of pixel weights
    for k = 1:M
        T(accepted(k,1), accepted(k,2)) = theta(k+1);
    end
    %fprintf('θ₀: %.3f \n', theta(1));
    figure; imagesc(T); colorbar; axis image; %heatmap
    end
    
    
    function[y_pred, test_error] = compute_error(A_test, theta_train, y_test) %computes the error rate of the function
    y_pred = sign(A_test * theta_train);
    test_error = mean(y_pred ~= y_test);
    end
    
    function[false_pos_rate, false_neg_rate, pos_y, neg_y, y_minus_sample, y_plus_sample] = compute_fpnr(y_test, y_pred) %computes false positive and false negative rate
    pos_y = 0;
    y_minus_sample = 0;
    y_plus_sample = 0;
    neg_y = 0;
    for k=1:length(y_test)
        if y_test(k) == 1
            y_plus_sample = y_plus_sample + 1; 
            if y_pred(k) == -1
                neg_y = neg_y + 1; %false positive counter
            end
        end
        if y_test(k) == -1
            y_minus_sample = y_minus_sample + 1; 
            if y_pred(k) == 1 %false negative counter
                pos_y = pos_y + 1;
            end
        end
    end
    false_pos_rate = pos_y/y_minus_sample * 100;
    false_neg_rate = neg_y/y_plus_sample * 100;
    %fprintf('%3.f', pos_y+neg_y);
    end

    function[F_train, F_test, R] = random_feature_map(A_train, A_test, M1) %random feature generator
    M0 = size(A_train, 2) - 1;
    rng(15);

    R = sign(randn(M1, M0));

    x_train = A_train(:,2:end);
    x_test = A_test(:, 2:end);

    F_train = max(x_train * R', 0);
    F_test = max(x_test * R', 0);
    end
