
clear all; clc;

% Step 1: Define System and Parameters %%%%%%%%%%%%%%%%%%%%%
% True system parameters (unknown to the algorithm)
A = [0 1; -1 -0.5]; % True system matrix A (unknown to the algorithm)
B = [0; 1];         % True input matrix B (unknown to the algorithm)

% Cost function matrices
Q = eye(2);         % State weighting matrix
R = 1;              % Control weighting matrix

% Time step for simulation
dt = 0.01;          % Time step size

% Simulation time
tspan = 0:dt:10;    % Simulation time span

% Initial state
x0 = [1; 0];        % Initial state vector

% Step 2: Define the System Dynamics %%%%%%%%%%%%%%%%%%
% System dynamics (unknown to the algorithm)
sys_dynamics = @(t, x, u) A * x + B * u;

% Step 3: Define the Cost Function  %%%%%%%%%%%%%%%%%%%%
% Cost function
cost_function = @(x, u) x' * Q * x + u' * R * u;

% Step 4: Initialize Policy Parameters  %%%%%%%%%%%%%%%%%
%Initial control policy (e.g., random or zero)
K = [0 0];          % Initial control gain matrix

% Policy gradient parameters
alpha = 0.01;       % Learning rate
num_episodes = 1000; % Number of episodes
exploration_noise = 0.1; % Exploration noise (e.g., Gaussian noise)

% Step 5: Policy Gradient Algorithm %%%%%%%%%%%%%%%%%%%%
% Main policy gradient loop
for episode = 1:num_episodes
    % Initialize the state
    x = x0;
    
    % Simulate the system for one episode
    total_cost = 0;
    for t = 1:length(tspan)
        % Apply control input with exploration noise
        u = -K * x + exploration_noise * randn; % Add exploration noise
        
        % Simulate the system for one time step
        x_dot = sys_dynamics(tspan(t), x, u);
        x_next = x + x_dot * dt;
        
        % Compute the cost
        total_cost = total_cost + cost_function(x, u) * dt;
        
        % Update the state
        x = x_next;
    end
    
    % Update the control policy (K) using policy gradient
    % Here, we use a simple gradient estimate based on the total cost
    K = K - alpha * total_cost * K;
end

% Step 6: Simulate the Final Control Policy  %%%%%%%%%%%%%%%%%%%%%
% Simulate the system with the final control policy
x = x0;             % Initial state
x_history = zeros(2, length(tspan)); % To store state history
u_history = zeros(1, length(tspan)); % To store control history

for t = 1:length(tspan)
    % Apply the final control policy
    u = -K * x;
    u_history(t) = u;
    
    % Simulate the system for one time step
    x_dot = sys_dynamics(tspan(t), x, u);
    x_next = x + x_dot * dt;
    
    % Update the state
    x = x_next;
    x_history(:, t) = x;
end

% Step 7: Plot the Results  %%%%%%%%%%%%%%%%%%%%%%%%%
% Plot the results
figure;
plot(tspan, x_history(1, :), 'r', tspan, x_history(2, :), 'b');
xlabel('Time');
ylabel('State');
legend('x_1', 'x_2');
title('System Response with Policy Gradient');

% Plot the control input
figure;
plot(tspan, u_history, 'g');
xlabel('Time');
ylabel('Control Input');
title('Control Input with Policy Gradient');