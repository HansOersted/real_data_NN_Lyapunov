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

%%
n_values = [10, 20, 30, 40];

errors = [error_10, error_20, error_30, error_40];

%%
figure;
hold on;

for j1 = 1:length(n_values)
    for j2 = 1:length(n_values)
        if j1 ~= j2 
            for i1 = 1:size(errors, 1)
                for i2 = 1:size(errors, 1)
                    plot([n_values(j1), n_values(j2)], [errors(i1, j1), errors(i2, j2)], '-o', 'LineWidth', 1.2, 'MarkerSize', 6);
                end
            end
        end
    end
end

xlabel('Numbers of traning samples', 'FontSize', 16);
ylabel('Error', 'FontSize', 16);
title('Generalization Error', 'FontSize', 18);

grid on;
hold off;
