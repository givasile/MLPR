D = 1:10:1000;

time = [];
for d = D
    X = rand(d, d);
    f = @() inv(X);
    time = [time timeit(f)];
end

plot(D, time, 'r-o');
hold on;
x = linspace(0, 1000, 10000);
time_per_command = 3 * 10e9;
plot(x, x.^3/time_per_command, 'b-');
plot(x, x.^2/time_per_command, 'g-');
plot(x, x.^2.*log(x)/time_per_command, 'c-');