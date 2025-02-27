clear
%%
A = [ 0.279545252782875  -0.034124554112530;
  -0.034124554112530   0.440942074913554];

lambda_val = 5;

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

n_training = 10;
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

%%
constraint_last_epoch_epsilon = [];
for i = 1 : n1
    for t = 1 : length
        % Extract Current Time Step Data
        de = derivative_training_sample(i).data(t, :)'; % Tracking error derivative
        dde = derivative_derivative_training_sample(i).data(t, :)'; % Second derivative
        constraint_clean_epsilon = dde' * A * de + de' * A * dde + lambda_val * de' * A * de;
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