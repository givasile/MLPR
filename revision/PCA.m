N = 1000;
K = 1;
s_noise = 10;

% generate data
X = rand([N,1])*3;
w = [1];
Y = X*w + randn(N,1)*s_noise;
XY = [X Y];

[V,E] = eig(cov(XY,1));

% sort eigenvectors
[E, id] = sort(diag(E), 1, 'descend');
V = V(:, id(1:K));

xy_bar = mean(XY, 1);
XY_projected = (XY - xy_bar) * V;

XY_reprojected = (XY_projected * V') + xy_bar;

% SE
tmp = sum((XY - XY_reprojected).^2,  2);
mean(tmp)
