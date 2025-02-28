clear
close all
%%
num_test = 250;
A = [  0.329307706919647   0.017873808679665;
   0.017873808679665   0.226444477927737 ];
epsilon = 0.0720;

lambda = 5;

%% Adopt csv data 
UR5_experiment = readtable('testing_data.csv');

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
%%

index_ = find(diff(time_csv)==0);

dq_ref_first(index_+1) = [];
time_csv(index_+1) = [];
dq_actual_first(index_+1) = [];
accc = diff(dq_ref_first);
index_start = 1;
N = 499;
for i = 1:1:N
    index_sample(i) = find(accc(index_start:end)<0,1,'first') + index_start-1;
    index_sample_end(i) = find(accc(index_sample(i):end)==0,1,'first') + index_sample(i) - 1;
    index_start = find(accc(index_sample_end(i):end)>0,1,'first') + index_sample_end(i) - 1;
end


%%

figure
plot(time_csv(1:end-1)-time_csv(1), diff(dq_ref_first))

figure
plot(time_csv-time_csv(1), dq_ref_first)
hold on
% plot(time_csv-time_csv(1), dq_actual_first)

for i=1:1:N
    plot(time_csv(index_sample(i))-time_csv(1),dq_ref_first(index_sample(i)),'rx')
    plot(time_csv(index_sample_end(i))-time_csv(1),dq_ref_first(index_sample_end(i)),'bx')
end

% figure
% plot(diff(time_csv))

%% Extract the nontrival idx
% 499 in total
nontrival_idx = [ index_sample.'  index_sample_end.' ];


%% Calculate the generalization error

generalization_index = [];
error_count = 0;
for i = 1 : num_test
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