% n = 2
error_2 = [ 0.453608247422680;
            0.288659793814433;
            0.731958762886598;
            0.484536082474227;
            0.340206185567010;
            0.597938144329897];

% n = 5
error_5 = [ 0.127659574468085;
            0.170212765957447;
            0.223404255319149;
            0.106382978723404;
            0.106382978723404;
            0.085106382978723];

% n = 10
error_10 = [   0.0787;
    0.022471910112360;
    0.101123595505618;
    0.033707865168539; 
    0.044943820224719; 
    0.056179775280899 ];

% n = 20
error_20 = [  0.063291139240506;
    0.101265822784810;
    0;
    0.037974683544304;
    0.151898734177215;
    0.012658227848101 ];

% n = 30
error_30 = [  0.086956521739130;
    0.057971014492754;
    0.144927536231884;
    0.057971014492754;
    0.086956521739130;
    0.057971014492754 ];

% n = 40
error_40 = [  0;
   0;
   0.033898305084746;
   0;
   0.050847457627119;
   0.101694915254237];

% n = 50
error_50 = [  0;
              0;
              0;
              0.020408163265306;
              0;
              0  ];

%%
n_values = [2, 5, 10, 20, 30, 40, 50];

errors = [error_2, error_5, error_10, error_20, error_30, error_40, error_50];

%%
figure;
hold on;

for j1 = 1:(length(n_values)-1)  
    j2 = j1 + 1; 
    for i1 = 1:size(errors, 1)
        for i2 = 1:size(errors, 1)
            plot([n_values(j1), n_values(j2)], [errors(i1, j1), errors(i2, j2)], '-o', 'LineWidth', 0.2, 'MarkerSize', 6, 'Color', 'b');
        end
    end
end

theoretical_1 = 4.6062 ./ n_values;
theoretical_2 = sqrt(4.6062 ./ n_values);

h1 = plot(n_values, theoretical_1, 'r-', 'LineWidth', 2); % 4.6062/n
h2 = plot(n_values, theoretical_2, 'g-', 'LineWidth', 2); % sqrt(4.6062/n)

legend([h1, h2], {'4.6062/n', 'sqrt(4.6062/n)'}, 'FontSize', 12);

xlim([0 50]);
ylim([0 1]);
xlabel('Numbers of training samples', 'FontSize', 16);
ylabel('Error', 'FontSize', 16);
title('Generalization Error', 'FontSize', 18);

grid on;
hold off;
