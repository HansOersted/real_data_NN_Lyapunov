clear
close all
clc
warning on
%% Highlight the important training parameters
lambda_val = 2;
num_epochs = 100;
learning_rate = 1e-3;

gamma = 1e-4;

%% Adopt csv data 

UR5_experiment_01 = readtable('robot_data_01rad.csv');
UR5_experiment_50 = readtable('robot_data_50rad.csv');

sample_time_csv = 0.002; % 500 Hz

time_csv_01 = UR5_experiment_01.timestamp;
time_csv_50 = UR5_experiment_50.timestamp;

q_ref_01 = UR5_experiment_01.target_q_5;
dq_ref_01 = UR5_experiment_01.target_qd_5;
ddq_ref_01 = UR5_experiment_01.target_qdd_5;
q_ref_50 = UR5_experiment_50.target_q_5;
dq_ref_50 = UR5_experiment_50.target_qd_5;
ddq_ref_50 = UR5_experiment_50.target_qdd_5;

q_actual_01 = UR5_experiment_01.actual_q_5;
dq_actual_01 = UR5_experiment_01.actual_qd_5;
q_actual_50 = UR5_experiment_50.actual_q_5;
dq_actual_50 = UR5_experiment_50.actual_qd_5;

e_csv_01 = q_actual_01 - q_ref_01;
de_csv_01 = dq_actual_01 - dq_ref_01;
e_csv_50 = q_actual_50 - q_ref_50;
de_csv_50 = dq_actual_50 - dq_ref_50;

dde_csv_01 = diff(de_csv_01) / sample_time_csv;  % lose 1 row
% dde_csv_01 = smooth(dde_csv_01);
ddde_csv_01 = diff(dde_csv_01) / sample_time_csv;  % lose 2 rows
% ddde_csv_01 = smooth(ddde_csv_01);
dde_csv_50 = diff(de_csv_50) / sample_time_csv;  % lose 1 row
% dde_csv_50 = smooth(dde_csv_50);
ddde_csv_50 = diff(dde_csv_50) / sample_time_csv;  % lose 2 rows
% ddde_csv_50 = smooth(ddde_csv_50);

interested_index_01 = [  1480:1493, ...
                         1979:1988, ...
                         2981:3063, ...
                         4968:4978, ...
                         5467:5478, ...
                         6468:6527, ...
                         8450:8464, ...
                         8949:8958, ...
                         9950:10012  ];

interested_index_50 = [  1329:1372, ...
                         1492:1572, ...
                         1690:1787, ...
                         4415:4457, ...
                         4578:4662, ...
                         4776:4857, ...
                         7497:7542, ...
                         7660:7742, ...
                         7858:7907   ];

de_interested_01 = de_csv_01(interested_index_01);
dde_interested_01 = dde_csv_01(interested_index_01);
ddde_interested_01 = ddde_csv_01(interested_index_01);

de_interested_50 = de_csv_50(interested_index_50);
dde_interested_50 = dde_csv_50(interested_index_50);
ddde_interested_50 = ddde_csv_50(interested_index_50);

de_interested = [de_interested_01; de_interested_50];
dde_interested = [dde_interested_01; dde_interested_50];
ddde_interested = [ddde_interested_01; ddde_interested_50];

n1 = 1; % The number of experiments
dimension = 1;

E_interested = [de_interested];
dE_interested = [dde_interested];

for i = 1 : n1
    derivative_training_sample(i).data = E_interested;
    derivative_derivative_training_sample(i).data = dE_interested;
end

length = size(de_interested,1);

%% Prepare for Training
h = 32; % Width of the hidden layer

% Define NN Weights
L1 = randn(h, dimension); % Input to hidden layer 1
b1 = zeros(h, 1);

L2 = randn(h, h); % Hidden layer 1 to hidden layer 2
b2 = zeros(h, 1);

L_out = randn(dimension * (dimension + 1)/2 , h); % Hidden layer to output
b_out = zeros(dimension * (dimension + 1)/2 , 1);

%% Training Loop
loss_history = zeros(num_epochs, 1);
A_history = [];
L_history = [];
constraint_history = zeros(num_epochs, 1);

constraint_first_epoch = []; % Store the constraint in the first epoch (debug)
constraint_last_epoch = []; % Store the constraint in the last epoch (debug)

for epoch = 1 : num_epochs
    total_loss_clean = 0;
    currentEpochs = epoch;
    
    % Initialize Gradients
    dL1 = zeros(size(L1));
    db1 = zeros(size(b1));
    dL2 = zeros(size(L2));
    db2 = zeros(size(b2));
    dL_out = zeros(size(L_out));
    db_out = zeros(size(b_out));

    for i = 1 : n1
        for t = 1 : length
            % Extract Current Time Step Data
            de = derivative_training_sample(i).data(t, :)'; % Tracking error derivative
            dde = derivative_derivative_training_sample(i).data(t, :)'; % Second derivative
            
            % Forward Pass (Using ReLU Instead of tanh)
            hidden1 = max(0, L1 * de + b1); % ReLU activation
            hidden2 = max(0, L2 * hidden1 + b2); % ReLU activation

            % Construct Lower Triangular L_pred
            L_flat = L_out * hidden2 + b_out;
            L_pred = zeros(dimension, dimension);
            L_pred(tril(true(dimension, dimension))) = L_flat;
            L_pred(logical(eye(dimension))) = log(1 + exp(L_pred(logical(eye(dimension))))); % Softplus activation
            if any(isinf(L_pred), 'all')  % check if L_pred contains Inf
                warning('L_pred contains Inf values!');
            end
            if any(isnan(L_pred), 'all')
                warning('L_pred contains NaN values!');
            end
            % if any(L_pred < 1e-8)
            %     warning('L_pred contains values smaller than 0.00000001!');
            % end

            % Constraint Computation
            A = L_pred * L_pred'; % Coefficient matrix
            if any(isinf(A), 'all')  % check if L_pred contains Inf
                warning('A contains Inf values!');
            end
            if any(isnan(A), 'all')
                warning('A contains NaN values!');
            end
            % if any(A < 1e-8)
            %     warning('A contains values smaller than 0.00000001!');
            % end
            constraint = dde' * A * de + de' * A * dde + lambda_val * de' * A * de + gamma;
            constraint_clean = constraint - gamma;

            % Store the constraint in the first epoch (debug)
            if epoch == 1
                constraint_first_epoch = [constraint_first_epoch, constraint_clean];
            end

            if epoch == num_epochs
                constraint_last_epoch = [constraint_last_epoch, constraint_clean];
            end

            % Loss Computation
            constraint_violation = max(0, constraint);
            loss_clean = max(0, constraint_clean);
            total_loss_clean = total_loss_clean + loss_clean;

            % Compute Gradient
            if constraint_violation > 0
                A1 = dde'; B1 = de;
                A2 = de'; B2 = dde;
                A3 = de'; B3 = de;
            
                grad_constraint = (A1' * B1' + B1 * A1) * L_pred ...
                                + (A2' * B2' + B2 * A2) * L_pred ...
                                + lambda_val * (A3' * B3' + B3 * A3) * L_pred;
            
                % Softplus gradient correction
                softplus_derivative = 1 ./ (1 + exp(-L_pred)); % Softplus derivative
                % grad_constraint = grad_constraint .* softplus_derivative; % gradient correction
                % gradient correction is corrected only on the diagnal
                grad_constraint(logical(eye(dimension))) = grad_constraint(logical(eye(dimension))) ...
                                          .* softplus_derivative(logical(eye(dimension)));
            else
                grad_constraint = zeros(size(L_pred));
            end
            
            % Lower triangular gradient
            grad_L_flat = grad_constraint(tril(true(dimension, dimension))); 
            
            % update the gradient
            dL_out = dL_out + grad_L_flat * hidden2';
            db_out = db_out + grad_L_flat;

            % Update Hidden Layers (ReLU Derivative)
            grad_hidden2 = (L_out' * grad_L_flat) .* (hidden2 > 0);
            dL2 = dL2 + grad_hidden2 * hidden1';
            db2 = db2 + grad_hidden2;
            
            grad_hidden1 = (L2' * grad_hidden2) .* (hidden1 > 0);
            dL1 = dL1 + grad_hidden1 * de';
            db1 = db1 + grad_hidden1;
        end
    end

    % Update Weights
    L1 = L1 - learning_rate * dL1 / (n1 * length);
    b1 = b1 - learning_rate * db1 / (n1 * length);
    L2 = L2 - learning_rate * dL2 / (n1 * length);
    b2 = b2 - learning_rate * db2 / (n1 * length);
    L_out = L_out - learning_rate * dL_out / (n1 * length);
    b_out = b_out - learning_rate * db_out / (n1 * length);
    
    A_history = [A_history; A];
    L_history = [L_history; L_pred];

    
    % Save History
    loss_history(epoch) = total_loss_clean;
    constraint_history(epoch) = constraint;
    
    % Debugging (Optional)
    if mod(epoch, 50) == 0
        fprintf('Epoch %d, Loss (clean): %.4f\n', epoch, total_loss_clean);
        L_pred
        A
    end
end

%% Plot Results
figure;
plot(loss_history, 'LineWidth', 2);
xlabel('Epoch');
ylabel('Loss');
title('Training Loss (Clean)');
grid on;

% figure;
% plot(constraint_history, 'LineWidth', 2);
% xlabel('Epoch');
% ylabel('Constraint');
% title('Constraint History');
% grid on;

%% Plot First and Last Epoch Constraints
figure;
plot(1:size(constraint_first_epoch,2), constraint_first_epoch, 'LineWidth', 2);
xlabel('Training Sample Index');
ylabel('Constraint Value');
title('Constraints in the First Epoch (Clean)');
grid on;

figure;
plot(1:size(constraint_last_epoch,2), constraint_last_epoch, 'LineWidth', 2);
xlabel('Training Sample Index');
ylabel('Constraint Value');
title('Constraints in the Last Epoch (Clean)');
grid on;



%%
de = 0:0.2:10;
A_plot = A;
eig(A)
Lyap = A_plot .* de .^ 2;
figure
plot(de,Lyap)
xlabel('$\dot{e}$', 'Interpreter', 'latex', 'FontSize', 18);
ylabel('$V(\dot{e})$', 'Interpreter', 'latex', 'FontSize', 18);
title(['Lyapunov Function ($\lambda = ' num2str(lambda_val) '$)'], 'Interpreter', 'latex', 'FontSize', 16);

