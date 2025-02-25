clear
close all

%% 
A = 0.275162800026349;
lambda = 2;
gamma = 0.01;

%% Adopt csv data 

UR5_experiment_01 = readtable('robot_data_01rad.csv');

sample_time_csv = 0.002; % 500 Hz

time_csv_01 = UR5_experiment_01.timestamp;

q_ref_01 = UR5_experiment_01.target_q_5;
dq_ref_01 = UR5_experiment_01.target_qd_5;
ddq_ref_01 = UR5_experiment_01.target_qdd_5;

q_actual_01 = UR5_experiment_01.actual_q_5;
dq_actual_01 = UR5_experiment_01.actual_qd_5;

e_csv_01 = q_actual_01 - q_ref_01;
de_csv_01 = dq_actual_01 - dq_ref_01;

dde_csv_01 = diff(de_csv_01) / sample_time_csv;  % lose 1 row
ddde_csv_01 = diff(dde_csv_01) / sample_time_csv;  % lose 2 rows


interested_index_01 = [  2981:3063  ];

de_interested_01 = de_csv_01(interested_index_01);
dde_interested_01 = dde_csv_01(interested_index_01);
ddde_interested_01 = ddde_csv_01(interested_index_01);


E_interested = [de_interested_01];
dE_interested = [dde_interested_01];


%%
% V = de' * A * de;
% 
% constraint = dde' * A * de + de' * A * dde + lambda * de' * A * de + gamma;