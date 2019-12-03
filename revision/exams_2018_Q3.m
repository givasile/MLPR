x = [10 30 50 70 90];
y = [-1.5 -0.5 0.7 1.3 0.9];

phi = [];
K = 1:100;
h = 100;
for xx = x
    tmp = [];
    for k = K
        tmp = [tmp RBF(xx, k, h)];
    end
    phi = [phi; tmp];
end


w = phi \ y.';


i = 0:0.1:100;
phi = [];
for xx = i
    tmp = [];
    for k = K
        tmp = [tmp RBF(xx, k, h)];
    end
    phi = [phi; tmp];
end
yy = phi*w;

figure();
plot(i, yy);
hold on;
plot(x,y, 'ro')

function y = RBF(x, k, h)
    y = exp(-((x-k)^2)/h^2); 
end



