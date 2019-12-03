%% Example Title
% Summary of example objective

%% Forward pass
% Inputs
w1 = [1 2; 3 4];
x = [3; 2];
b = 1;
w2 = [1; 1];
y1 = 3;

% Graph: Forward pass
y = w1*x;          % linear
z = y + b;         % bias
a = z; a(a<0) = 0; % ReLU
y1_hat = w2.'*a;   % last layer
E = y1_hat - y1;   % error
SE = E.^2;         % square error


%% Reverse Graph
% Compute Jacobian Matrices

E_bar = 2*E;
y1_hat_bar = E_bar;
y1_bar = - E_bar;
a_bar = w2*y1_hat_bar;
w2_bar = y1_hat_bar*a.';
z_bar = a; z_bar(a>=0) = 1; z_bar(a<0) = 0; z_bar = z_bar.*a_bar;
y_bar = z_bar;
b_bar = z_bar;
w1_bar = y_bar*x.';
x_bar = w1.'*y_bar;


%% gradient apply
t = 0.01;
w1 = w1 - t*w1_bar;
w2 = w2 - t*w2_bar.';
b = b - t*b_bar;