% Script illustrating the equivalence of PCA and truncated SVD.
% The generated data follows X2 = X1 + e, where X1~U(0,1), e ~ N(0,s_noise)
% Both methods reduce the m


% generate data [X1,X2] where X2=X1+e)
N = 100;
K = 1;
s_noise = .3;
X = rand([N,1])*3;
w = .3;
Y = X*w + randn(N,1)*s_noise;
X = [X Y];
X_bar = mean(X, 1);

% apply PCA
V1 = PCA_with_eigenvalues(X);

% apply SVD
[U2, S2, V2] = svd(X - X_bar, 0);

% crop K<D dimensions
V1 = V1(:,K);
V2 = V2(:,K);
disp(sum(sum(V1-V2<1.0e-10)) == 0)

% reproject points
X_repr_1 = (X-X_bar)*(V1*V1') + X_bar;
X_repr_2 = (X-X_bar)*(V2*V2') + X_bar;

% SE
tmp = sum((X - X_repr_1).^2,  2);
mean(tmp)

figure()
plot(X(:,1), X(:,2), 'ro')
hold on
plot(X_repr_1(:,1), X_repr_1(:,2), 'bo')
plot(X_repr_2(:,1), X_repr_2(:,2), 'gx')


function [W] = PCA_with_eigenvalues(X)
    % find eigenvalues
    [W, E] = eig(cov(X,1));
    disp(E)
    
    % sort eigenvectors based on descending order of eigenvalues
    [~, id] = sort(diag(E), 1, 'descend');
    W = W(:, id(:));
end