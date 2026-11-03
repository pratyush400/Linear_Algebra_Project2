[imgs, labels] = readMNIST('train-images-idx3-ubyte','train-labels-idx1-ubyte',5000,0);
function [matrix_count, accepted] = extractor(imgs) 
[num_rows, num_cols, total_images] = size(imgs);
matrix_count = zeros(width(imgs),height(imgs));
for n=1:total_images
    for i=1:num_rows
        for j=1:num_cols
            if imgs(i,j,n)> 0
                matrix_count(i,j) = matrix_count(i,j) + 1;
            end
        end
    end
end
[count_rows, count_cols, count_tot_img] = size(matrix_count);
accepted = [];
for i=1:count_rows
    for j=1:count_cols
        if matrix_count(i,j) >= 600
            accepted(end+1,:) = [i,j,matrix_count(i,j)];
        end
    end
end
end

y = zeros(length(labels),1);

for i=1:length(labels)
    if labels(i) > 0
        y(i) = -1;
    else y(i) = 1;
    end
end

disp(size(accepted))
M = size(accepted, 1); 
A = zeros(5000, M+1);


A(:,1) = 1;


for n= 1:total_images
    for k=1:M
        target_row = accepted(k, 1);
        target_col = accepted(k, 2);
        A(n,k+1) = imgs(target_row,target_col,n);
 
    end
end

theta = zeros(size(y));

theta = A \ y;


theta0 = theta(1);
T = zeros(20,20);
for k = 1:M
    T(accepted(k,1), accepted(k,2)) = theta(k+1);
end
imagesc(T); colorbar; axis image;



[imgs_test, labels_test] = readMNIST('t10k-images-idx3-ubyte','t10k-labels-idx1-ubyte',5000,0);

[test_matrixcount, test_accepted] = extractor(imgs_test);


             
