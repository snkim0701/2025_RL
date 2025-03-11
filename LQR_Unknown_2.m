

clear all; clc;
% True system parameters (unknown to the algorithm)
A = [0 1; -1 -0.5]; % Unknown system matrix A
B_true = [0; 1];    % True input matrix B (unknown to the algorithm)

% Cost function matrices
Q = eye(2);         % State weighting matrix
R = 1;              % Control weighting matrix

% Time interval for integral reward
T = 0.1;            % Time horizon for integral reward

% Initial state
x0 = [1; 0];        % Initial state vector

% Simulation time
tspan = [0 10];     % Simulation time span

% System dynamics (unknown to the algorithm)
sys_dynamics = @(t, x, u) A * x + B_true * u;

% Cost function
cost_function = @(x, u) x' * Q * x + u' * R * u;

% Initial control policy (e.g., random or zero)
K = [0 0];          % Initial control gain matrix

% Initial value function parameters (e.g., zero)
P = zeros(2, 2);    % Initial value function matrix

% Collect input-output data to estimate B
N = 1000;           % Increase the number of data points for better estimation
t_data = linspace(0, 1, N); % Time points for data collection
x_data = zeros(2, N); % State data
u_data = zeros(1, N); % Input data

% Simulate the system with random inputs to collect data
x_current = x0;     % Initialize the state
for i = 1:N
    u_data(i) = randn; % Random input
    [~, x] = ode45(@(t, x) sys_dynamics(t, x, u_data(i)), [0 T], x_current);
    x_data(:, i) = x(end, :)';
    x_current = x(end, :)'; % Update the current state
end

% Estimate B using least squares
X_dot = diff(x_data, 1, 2) / T; % Approximate derivative of x
X = x_data(:, 1:end-1);         % State data
U = u_data(1:end-1);            % Input data
B_est = (X_dot - A * X) / U;    % Least squares estimate of B

% IRL parameters
alpha = 0.01;       % Reduce the learning rate for stability
num_iterations = 500; % Increase the number of iterations for convergence

% Main IRL loop
x_current = x0;     % Initialize the state
for iter = 1:num_iterations
    % Simulate the system for one time interval
    [t, x] = ode45(@(t, x) sys_dynamics(t, x, -K * x), [0 T], x_current);
    
    % Compute the integral reward
    integral_reward = 0;
    for i = 1:length(t)-1
        u = -K * x(i, :)';
        integral_reward = integral_reward + cost_function(x(i, :)', u) * (t(i+1) - t(i));
    end
    
    % Update the value function parameters (P)
    delta = integral_reward + x(end, :) * P * x(end, :)' - x(1, :) * P * x(1, :)';
    P = P + alpha * delta;
    
    % Update the control policy (K) using the estimated B
    K = inv(R) * B_est' * P;
    
    % Update the current state for the next iteration
    x_current = x(end, :)';
end

% Simulate the system with the final control policy
[t, x] = ode45(@(t, x) sys_dynamics(t, x, -K * x), tspan, x0);

% Plot the results
figure;
plot(t, x(:, 1), 'r', t, x(:, 2), 'b');
xlabel('Time');
ylabel('State');
legend('x_1', 'x_2');
title('System Response with Optimal Control');

% Verify the initial state
disp('Initial state:');
disp(x(1, :)');
