clear
close all
%%
n_training = 99;
A = [  0.394609083791888  -0.157402130261373
  -0.157402130261373   0.527967082668835 ];
epsilon = 0.0806;

data_num = 2; % plot the constraint
lambda = 5;

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

length_ = length(training_index);

%%
constraint_last_epoch_epsilon = [];
for i = 1 : n1
    for t = 1 : length_
        % Extract Current Time Step Data
        de = derivative_training_sample(i).data(t, :)'; % Tracking error derivative
        dde = derivative_derivative_training_sample(i).data(t, :)'; % Second derivative
        constraint_clean_epsilon = dde' * A * de + de' * A * dde + lambda * de' * A * de;
        constraint_last_epoch_epsilon = [constraint_last_epoch_epsilon, constraint_clean_epsilon];
    end
end

%%
figure;
plot(1:size(constraint_last_epoch_epsilon,2), constraint_last_epoch_epsilon, 'LineWidth', 2);
xlabel('Training Sample Index');
ylabel('Constraint Value');
title('Constraints in the Last Epoch (Clean)');
grid on;

%%

verification_index = nontrival_idx(data_num,1) : nontrival_idx(data_num,2);

de_verification_first = de_csv_first(verification_index);
de_verification_second = de_csv_second(verification_index);

dde_verification_first = dde_csv_first(verification_index);
dde_verification_second = dde_csv_second(verification_index);

dE = [de_verification_first  de_verification_second];
ddE = [dde_verification_first  dde_verification_second];

V = [];
constraint = [];
time = [];

for i = 1 : length(verification_index) 
    de_verification = dE(i,:)';
    dde_verification = ddE(i,:)';
    V = [ V de_verification' * A * de_verification ];
    constraint = [ constraint dde_verification' * A * de_verification + de_verification' * A * dde_verification + lambda * de_verification' * A * de_verification - epsilon];
    time = [ time time_csv(verification_index(i)) ];
end

% figure
% subplot(2,1,1)
% plot(1 : length(verification_index) ,V)
% subplot(2,1,2)
% plot(1 : length(verification_index) ,constraint)
% 
% figure
% plot(1 : length(verification_index) ,dq_ref_first(verification_index))
% 
% figure
% plot(1 : length(verification_index) ,dq_actual_first(verification_index))

%% V and constraint

delay_fix = 3;  % UR5 has 2 steps delay
time = time(delay_fix:end) - time(delay_fix);
V = V(delay_fix:end);
constraint = constraint(delay_fix:end);

figure
hold on
grid on

% 获取当前坐标轴句柄
ax = gca;

% 设置左侧 y 轴
yyaxis left
h1 = plot(time, V, "Color", '#0072BD', 'LineWidth', 2); % 蓝色线，左 y 轴
ylabel('V', 'FontSize', 20, 'Interpreter', 'latex', "Color", '#0072BD'); % 设置左 y 轴标签
ylim([-0.0001, max(V)+abs(max(V))*0.1]); % 调整左 y 轴范围
ax.YColor = '#0072BD'; % 设置左侧 y 轴刻度颜色

% 添加蓝色虚线表示 V = 0，但不添加到图例
yline(0, '--b', 'LineWidth', 2, 'HandleVisibility', 'off');

% 设置右侧 y 轴
yyaxis right
h2 = plot(time, constraint, "Color", '#A2142F', 'LineWidth', 2); % 红色线，右 y 轴
ylabel('Constraint', 'FontSize', 20, 'Interpreter', 'latex', "Color", '#A2142F'); % 设置右 y 轴标签
ylim([min(constraint)-abs(min(constraint))*0.1, 0.01]); % 调整右 y 轴范围
ax.YColor = '#A2142F'; % 设置右侧 y 轴刻度颜色

% 添加红色虚线表示 constraint = 0，但不添加到图例
yline(0, '--r', 'LineWidth', 2, 'HandleVisibility', 'off');


% 设置 x 轴标签
xlabel('Time', 'FontSize', 18, 'Interpreter', 'latex');

% 设置标题
title('Lyapunov and Constraint', 'FontSize', 22, 'Interpreter', 'latex');

% 添加图例，只包含 V 和 Constraint
legend([h1, h2], {'Lyapunov function', 'Constraint (bounded noise)'}, 'Location', 'best', 'FontSize', 14);

hold off

%% Calculate the generalization error

generalization_index = [];
error_count = 0;
num_test = size(nontrival_idx,1) - n_training;
for i = (n_training + 1) : size(nontrival_idx,1)
    generalization_index = (nontrival_idx(i,1) + 2) : nontrival_idx(i,2);
    constraint = [];
    for j = 1 : length(generalization_index)
        de_generalization_first = de_csv_first(generalization_index(j));
        de_generalization_second = de_csv_second(generalization_index(j));
        de_generalization = [de_generalization_first; de_generalization_second];
    
        dde_generalization_first = dde_csv_first(generalization_index(j));
        dde_generalization_second = dde_csv_second(generalization_index(j));
        dde_generalization = [dde_generalization_first; dde_generalization_second];
    
        constraint = [ constraint dde_generalization' * A * de_generalization + de_generalization' * A * dde_generalization + lambda * de_generalization' * A * de_generalization - epsilon];
    end
    if any(constraint > 0)
        error_count = error_count + 1;
    end
end

generalization_error = error_count/num_test