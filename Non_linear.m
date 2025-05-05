%% Model-based Optimal controller

clear all;clc;
tic

% Problem setup
N = 50; T = 10;
t = linspace(0, T, N);
x = sdpvar(2, N); % States: x1 = position, x2 = velocity
u = sdpvar(1, N-1); % Control input

% Nonlinear dynamics (example: Duffing oscillator)
f = @(x, u) [x(2); -x(1) + u - x(2)^3];

% Constraints and objective
constraints = [];
objective = 0;
for k = 1:N-1
    x_next = x(:,k) + (T/N)*f(x(:,k), u(k));
    constraints = [constraints, x(:,k+1) == x_next];
    constraints = [constraints, -1 <= u(k) <= 1]; % Control bounds
    objective = objective + x(1,k)^2 + x(2,k)^2 + u(k)^2; % Quadratic cost
end
constraints = [constraints, x(:,1) == [1; 0]]; % Initial condition

% Solve
options = sdpsettings('solver', 'fmincon'); % Fallback if IPOPT missing
optimize(constraints, objective, options);

% Extract and plot
u_opt = value(u);
x_opt = value(x);

% Plot control
figure;
stairs(t(1:end-1), u_opt, 'r', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('u(t)');
title('Optimal Control Input');
grid on;

% Plot states
figure;
plot(t, x_opt(1,:), 'b', t, x_opt(2,:), 'r--', 'LineWidth', 2);
legend('x_1 (Position)', 'x_2 (Velocity)');
xlabel('Time (s)');
title('State Trajectories');
grid on;

toc



%% RL without Experience Replay
clear all; clc; 
tic
% Nonlinear system dynamics
system_dynamics = @(t, x, u) [x(2); -x(1) - x(2)^3 + u];

% State constraints
state_constraints = @(x) x(1)^2 + x(2)^2 <= 10;

% Control constraints
control_constraints = @(u) abs(u) <= 1;

% Goal state
x_goal = [0; 0];
goal_tolerance = 1e-3; % Tolerance for reaching the goal

% Critic (value function) parameters
V = @(x) x' * x; % Initial value function

% Actor (policy) parameters
policy = @(x) -x(1); % Initial policy

% Learning parameters
alpha_critic = 0.01; % Learning rate for critic
alpha_actor = 0.01;  % Learning rate for actor
gamma = 0.99;        % Discount factor

% Simulation time
dt = 0.01;          % Time step size
tspan = 0:dt:10;    % Simulation time span

% Initial state
x = [1; 0];         % Initial state vector
X = [ ];
% Main real-time loop
for t = 1:length(tspan)
    % Check termination condition (goal reached)
    if norm(x - x_goal) < goal_tolerance
        disp('Goal reached!');
        break;
    end
    
    % Choose an action using the current policy
    u = policy(x);
    
    % Ensure the control input satisfies the constraints
    u = min(max(u, -1), 1);
    
    % Simulate the system for one time step
    x_dot = system_dynamics(tspan(t), x, u);
    x_next = x + x_dot * dt;
    
    % Check state constraints
    if ~state_constraints(x_next)
        disp('State constraint violated!');
        break;
    end
    
    % Compute the reward
    r = -x' * x - u^2; % Reward (negative cost)
    
    % Critic update
    delta = r + gamma * V(x_next) - V(x);
    V = @(x) V(x) + alpha_critic * delta;
    
    % Actor update
    policy = @(x) policy(x) + alpha_actor * delta * u;
    
    % Update the state
    x = x_next;
    X=[X x];
end

figure(2);
plot(X(1,:)); hold on; grid on
plot(X(2,:))
toc



%% %%% Corrected Experience Replay Version
clear all; clc; 
tic

% System parameters
system_dynamics = @(t, x, u) [x(2); -x(1) - x(2)^3 + u];
state_constraints = @(x) x(1)^2 + x(2)^2 <= 10;
control_constraints = @(u) abs(u) <= 1;
x_goal = [0; 0];
goal_tolerance = 1e-3;

% Value function and policy definitions
V = @(x, P) x' * P * x;  % Quadratic value function
policy = @(x, K) -K * x;   % Linear policy (now properly defined)

% Initialize parameters
P = [1 0.1; 0.1 1]; % Initial P matrix
K = [1 0.5];        % Initial policy gain matrix

% Experience Replay parameters
buffer_capacity = 5000;
batch_size = 128;
min_buffer_size = 500;

% Learning parameters
alpha_critic = 0.005;
alpha_actor = 0.005;
gamma = 0.99;
exploration_rate = 0.3;
exploration_decay = 0.995;

% Simulation setup
dt = 0.01;
tspan = 0:dt:10;
x = [1; 0];
X = [];
replay_buffer = struct('x', {}, 'u', {}, 'r', {}, 'x_next', {});

for t = 1:length(tspan)
    if norm(x - x_goal) < goal_tolerance
        disp('Goal reached!');
        break;
    end
    
    % Decaying exploration
    current_exploration = exploration_rate * exploration_decay^t;
    
    % Action selection with exploration
    if rand() < current_exploration
        u = 2*rand() - 1; % Random action between -1 and 1
    else
        u = policy(x, K); % Now this will work correctly
    end
    u = min(max(u, -1), 1); % Enforce control constraints
    
    % System simulation
    x_dot = system_dynamics(tspan(t), x, u);
    x_next = x + x_dot * dt;
    
    if ~state_constraints(x_next)
        disp('State constraint violated!');
        break;
    end
    
    % Reward calculation
    r = -x'*x - 0.1*u^2 - 10*(norm(x-x_goal)<0.1);
    
    % Store experience
    if length(replay_buffer) < buffer_capacity
        replay_buffer(end+1) = struct('x', x, 'u', u, 'r', r, 'x_next', x_next);
    else
        replay_buffer(randi(buffer_capacity)) = struct('x', x, 'u', u, 'r', r, 'x_next', x_next);
    end
    
    x = x_next;
    X = [X x];
    
    % Experience replay update
    if length(replay_buffer) >= min_buffer_size
        batch_indices = randperm(length(replay_buffer), batch_size);
        batch = replay_buffer(batch_indices);
        
        % Batch updates
        delta_P = zeros(size(P));
        delta_K = zeros(size(K));
        
        for i = 1:batch_size
            exp = batch(i);
            delta = exp.r + gamma*V(exp.x_next, P) - V(exp.x, P);
            
            % Critic update
            delta_P = delta_P + alpha_critic * delta * (exp.x * exp.x');
            
            % Actor update
            delta_K = delta_K + alpha_actor * delta * exp.u * exp.x';
        end
        
        % Apply updates
        P = P + delta_P/batch_size;
        K = K + delta_K/batch_size;
        
        % Ensure symmetry of P
        P = 0.5*(P + P');
    end
end

% Plotting
figure(1);
plot(X(1,:)); hold on; grid on
plot(X(2,:))
xlabel('Time steps');
ylabel('State values');
legend('x1', 'x2');
title('Corrected Experience Replay Performance');

toc