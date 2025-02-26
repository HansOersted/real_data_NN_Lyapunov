clear
close all

%% 
A = 0.035905175142147;
lambda = 10;
% gamma = 0.01;

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


interested_index_01 = [  1480:1493  ];

time_interested_01 = time_csv_01(interested_index_01) - time_csv_01(interested_index_01(1));
de_interested_01 = de_csv_01(interested_index_01);
dde_interested_01 = dde_csv_01(interested_index_01);
ddde_interested_01 = ddde_csv_01(interested_index_01);


%%
V = de_interested_01 .* A .* de_interested_01;
constraint = dde_interested_01 .* A .* de_interested_01 + de_interested_01 .* A .* dde_interested_01 + lambda * de_interested_01 .* A .* de_interested_01;
figure
subplot(2,1,1)
plot(time_interested_01,V)
subplot(2,1,2)
plot(time_interested_01,constraint)

%% phase diagram

figure
hold on

plot(de_interested_01, dde_interested_01, '-o', ...
    'Color', '#0072BD', 'LineWidth', 3, ...   
    'MarkerSize', 10, ...                     
    'MarkerEdgeColor', '#A2142F', ...        
    'MarkerFaceColor', 'w', ...              
    'Marker', 'o');                    

xlabel('$$\dot{e}$$', 'Interpreter', 'latex', 'FontSize', 22);
ylabel('$$\ddot{e}$$', 'Interpreter', 'latex', 'FontSize', 22);

title('Phase Diagram ($$\dot{e}$$ vs. $$\ddot{e}$$)', 'Interpreter', 'latex', 'FontSize', 22);

grid on;

for i = 1:6
    text(de_interested_01(i) + 0.004, dde_interested_01(i) + 2, sprintf('%i', i), ...
        'FontSize', 18, 'FontWeight', 'bold', 'Color', 'k', 'HorizontalAlignment', 'left');
end

i = 7;
text(de_interested_01(i), dde_interested_01(i) + 10, sprintf('%i', i), ...
        'FontSize', 18, 'FontWeight', 'bold', 'Color', 'k', 'HorizontalAlignment', 'left');

i = length(de_interested_01);
text(de_interested_01(i) + 0.002, dde_interested_01(i) + 2, sprintf('%i', i), ...
        'FontSize', 18, 'FontWeight', 'bold', 'Color', 'k', 'HorizontalAlignment', 'left');

xline(0, '--k', 'LineWidth', 1.5); 
yline(0, '--k', 'LineWidth', 1.5);

hold off


%% V and constraint


