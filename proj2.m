[imgs, labels] = readMNIST('train-images-idx3-ubyte','train-labels-idx1-ubyte',5000,0);
[matrix_count, accepted] = extractor(imgs);
[y_train, A_train] = design_matrix_builder(imgs, labels, accepted);
[theta_train, T_train] = least_squares(y_train, A_train, accepted, size(imgs,1), size(imgs,2));

[imgs_test, labels_test] = readMNIST('t10k-images-idx3-ubyte','t10k-labels-idx1-ubyte',5000,0);
[y_test, A_test] = design_matrix_builder(imgs_test, labels_test, accepted);  

y_pred = sign(A_test * theta_train);
test_error = mean(y_pred ~= y_test);
fprintf('Test error: %.2f%%\n', 100*test_error);
pos_y = 0;
y_minus_sample = 0;
y_plus_sample = 0;
neg_y = 0;
for k=1:length(y_test)
    if y_pred == 1
        y_plus_sample = y_plus_sample + 1;
    if y_test(k) == -1 
        pos_y = pos_y +1 ;
    end
    end
    if y_pred == -1
        y_minus_sample = y_minus_sample + 1;
    if y_test(k) == 1
        neg_y = neg_y + 1;
    end
    end
end
    false_pos_rate = pos_y/length(y_pred);
    false_neg_rate = neg_y/length(y_pred);
fprintf('Test error: %.2f%%\n', 100*test_error);


function [matrix_count, accepted] = extractor(imgs)
[num_rows, num_cols, total_images] = size(imgs);
matrix_count = zeros(num_rows, num_cols);
for n = 1:total_images
    for i = 1:num_rows
        for j = 1:num_cols
            if imgs(i,j,n) > 0
                matrix_count(i,j) = matrix_count(i,j) + 1;
            end
        end
    end
end
accepted = [];
for i = 1:num_rows
    for j = 1:num_cols
        if matrix_count(i,j) >= 600
            accepted(end+1,:) = [i,j,matrix_count(i,j)];
        end
    end
end
end

function [y, A] = design_matrix_builder(imgs, labels, accepted)
total_images = size(imgs, 3);
y = ones(length(labels),1);
y(labels > 0) = -1;
M = size(accepted, 1);
A = zeros(total_images, M+1);
A(:,1) = 1;
for n = 1:total_images
    for k = 1:M
        A(n,k+1) = imgs(accepted(k,1), accepted(k,2), n);
    end
end
end

function [theta, T] = least_squares(y, A, accepted, num_rows, num_cols)
theta = A \ y;
M = size(accepted, 1);
T = zeros(num_rows, num_cols);
for k = 1:M
    T(accepted(k,1), accepted(k,2)) = theta(k+1);
end
figure; imagesc(T); colorbar; axis image;
end
