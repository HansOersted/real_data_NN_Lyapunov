clear
close all
clc
warning on
%% Highlight the important training parameters
n_training = 10;

lambda_val = 5;
num_epochs = 2000;
learning_rate = 1e-3;
gamma = 1e-4;

%% Adopt csv data 
UR5_experiment = readtable('final_data.csv');

sample_time_csv = 0.002; % 500 Hz

time_csv = UR5_experiment.timestamp;

q_ref_first = UR5_experiment.target_q_3;
dq_ref_first = UR5_experiment.target_qd_3;
ddq_ref_first = UR5_experiment.target_qdd_3;
q_ref_second = UR5_experiment.target_q_5;
dq_ref_second = UR5_experiment.target_qd_5;
ddq_ref_second = UR5_experiment.target_qdd_5;

q_actual_first = UR5_experiment.actual_q_3;
dq_actual_first = UR5_experiment.actual_qd_3;
q_actual_second = UR5_experiment.actual_q_5;
dq_actual_second = UR5_experiment.actual_qd_5;

e_csv_first = q_actual_first - q_ref_first;
de_csv_first = dq_actual_first - dq_ref_first;
e_csv_second = q_actual_second - q_ref_second;
de_csv_second = dq_actual_second - dq_ref_second;

dde_csv_first = diff(e_csv_first) / sample_time_csv;  % lose 1 row
% dde_csv_first = smooth(dde_csv_first);
ddde_csv_first = diff(dde_csv_first) / sample_time_csv;  % lose 2 rows
% ddde_csv_first = smooth(ddde_csv_first);
dde_csv_second = diff(e_csv_second) / sample_time_csv;  % lose 1 row
% dde_csv_second = smooth(dde_csv_second);
ddde_csv_second = diff(dde_csv_second) / sample_time_csv;  % lose 2 rows
% ddde_csv_second = smooth(ddde_csv_second);

%% Extract the nontrival idx
% 99 in total
nontrival_idx = [  644     665   ;
                   845     869   ;
                   1045    1052  ;
                   1245    1264  ;
                   1446    1462  ;
                   1646    1667  ;
                   1846    1857  ;
                   2046    2070  ;
                   2247    2256  ;
                   2447    2458  ;
                   2647    2669  ;
                   2848    2860  ;
                   3048    3066  ;
                   3248    3269  ;
                   3448    3459  ;
                   3649    3660  ;
                   3849    3871  ;
                   4049    4073  ;
                   4250    4269  ;
                   4450    4464  ;
                   4648    4672  ;
                   4848    4870  ;
                   5049    5072  ;
                   5249    5263  ;
                   5449    5467  ; 
                   5650    5671  ;
                   5850    5867  ;
                   6050    6069  ;
                   6251    6265  ;
                   6451    6471  ;
                   6651    6668  ;
                   6852    6873  ;
                   7051    7062  ;
                   7251    7276  ;
                   7451    7471  ;
                   7652    7657  ;
                   7852    7875  ;
                   8052    8072  ;
                   8253    8273  ;
                   8453    8469  ;
                   8653    8676  ;
                   8853    8860  ;
                   9054    9076  ;
                   9254    9269  ;
                   9454    9469  ;
                   9653    9675  ;
                   9853    9862  ;
                   10053   10070 ;
                   10254   10275 ;
                   10454   10477 ;
                   10654   10663 ;
                   10854   10869 ;
                   11055   11072 ;
                   11255   11276 ;
                   11455   11461 ;
                   11656   11679 ;
                   11856   11861 ;
                   12055   12072 ;
                   12256   12277 ;
                   12456   12472 ;
                   12656   12673 ;
                   12856   12880 ;
                   13057   13071 ;
                   13257   13282 ;
                   13455   13473 ;
                   13658   13666 ;
                   13858   13882 ;
                   14058   14073 ;
                   14259   14282 ;
                   14459   14474 ;
                   14659   14679 ;
                   14859   14874 ;
                   15060   15067 ;
                   15260   15274 ;
                   15460   15468 ;
                   15661   15682 ;
                   15861   15874 ;
                   16061   16081 ;
                   16262   16267 ;
                   16462   16477 ;
                   16662   16681 ;
                   16862   16865 ;
                   17063   17083 ;
                   17263   17278 ;
                   17463   17485 ;
                   17663   17674 ;
                   17864   17883 ;
                   18064   18087 ;
                   18264   18278 ;
                   18465   18470 ;
                   18664   18685 ;
                   18864   18887 ;
                   19065   19083 ;
                   19265   19279 ;
                   19465   19481 ;
                   19666   19675 ;
                   19866   19889 ;
                   20066   20086 ;
                   20266   20286   ];


%% Pick the index for training
training_index = [];

for i = 1 : n_training
    training_index = [ training_index nontrival_idx(i,1) : nontrival_idx(i,2) ];
end


%%
de_training_first = de_csv_first(training_index);
dde_training_first = dde_csv_first(training_index);
ddde_training_first = ddde_csv_first(training_index);
de_training_second = de_csv_second(training_index);
dde_training_second = dde_csv_second(training_index);
ddde_training_second = ddde_csv_second(training_index);


de_training = [de_training_first  de_training_second];
dde_training = [dde_training_first  dde_training_second];
ddde_training = [ddde_training_first  ddde_training_second];

n1 = 1; % The number of experiments
dimension = 2;

E_interested = [de_training];
dE_interested = [dde_training];

for i = 1 : n1
    derivative_training_sample(i).data = E_interested;
    derivative_derivative_training_sample(i).data = dE_interested;
end

length = length(training_index);

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
[e,de] = meshgrid(-20:1:20,-20:1:20);

A_plot = A;

eig(A)

Lyap = zeros(size(e));

for i = size(e,1):-1:1
    for j = size(e,1):-1:1
        Lyap(i,j) = [e(i,j)  de(i,j)] * A_plot * [e(i,j) ; de(i,j)];
    end
end
figure
surf(e,de,Lyap)
xlabel('$e$', 'Interpreter', 'latex', 'FontSize', 18);
ylabel('$\dot{e}$', 'Interpreter', 'latex', 'FontSize', 18);
zlabel('$V(e, \dot{e})$', 'Interpreter', 'latex', 'FontSize', 16);
title(['Lyapunov Function ($\lambda = ' num2str(lambda_val) '$)'], 'Interpreter', 'latex', 'FontSize', 16);