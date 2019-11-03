%% measure complexity of reversing a matrix
D = 1:10:1300;
time_per_command = 3 * 10e9;

time = [];
for d = D
    X = rand(d, d);
    f = @() inv(X);
    time = [time timeit(f)];
end

figure();
plot(D, time, 'r-o', 'DisplayName', 'A^{-1}');
hold on;
x = linspace(0, D(end), 10000);
plot(x, x.^3/time_per_command, 'b-', 'DisplayName', 'x^3');
plot(x, x.^2/time_per_command, 'g-', 'DisplayName', 'x^2');
plot(x, x.^2.*log(x)/time_per_command, 'c-', 'DisplayName', 'x^2 log(x)');
legend()
title('Computational complexity of inverting a matrix');
xlabel('Size of matrix')
ylabel('time (s)')
hold off;

%% measure complexity of A*B
D = 1:10:1000;

time = [];
for d = D
    X = rand(d, d);
    f = @() inv(X);
    time = [time timeit(f)];
end

hold on;
plot(D, time, 'r-o', 'DisplayName', 'A^{-1}');
x = linspace(0, 1000, 10000);
time_per_command = 3 * 10e9;
plot(x, x.^3/time_per_command, 'b-', 'DisplayName', 'x^3');
plot(x, x.^2/time_per_command, 'g-', 'DisplayName', 'x^2');
plot(x, x.^2.*log(x)/time_per_command, 'c-', 'DisplayName', 'x^2 log(x)');
legend()
hold off;

%% computational complexity of fitting a bayes binary classifier
D = 1:10:1000;
N = 500;

time = [];
for d = D
    X = rand(N, d);
    Y = randi([0 1], N, 1);

    X_1 = X( Y == 1, : );
    shift_1 = 10;
    X_1 = X_1 + shift_1;

    X_0 = X( Y == 0, : );
    shift_0 = -10;
    X_0 = X_0 + shift_0;

    f = @() fit_gaussian(X);
    % m_1, S_1 = fit_gaussian(X_1);
    % m_2, S_2 = fit_gaussian(X_2);

    time = [time timeit(f)];
end

hold on;
plot(D, time, 'r-o', 'DisplayName', 'A^{-1}');
x = linspace(0, 1000, 10000);
time_per_command = 3 * 10e9;
plot(x, x.^3/time_per_command, 'b-', 'DisplayName', 'x^3');
plot(x, x.^2/time_per_command, 'g-', 'DisplayName', 'x^2');
plot(x, x.^2.*log(x)/time_per_command, 'c-', 'DisplayName', 'x^2 log(x)');
legend()
hold off;

