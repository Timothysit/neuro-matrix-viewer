%% Generate example matlab data format 

% Define parameters of the simulation 
num_time_points = 400;
num_cells = 200;
num_trials = 20;
num_increase_activity_trials = 20;
num_decrease_activity_trials = 20;

exp_start_time = -10;
exp_end_time = 30;
exp_time_coord = linspace(exp_start_time, exp_end_time, num_time_points);

% Make the matrices

baseline_noise = normrnd(0, 1, num_cells, num_time_points);
increase_activity = linspace(0, 1, num_time_points) ...
    +  normrnd(0, 1, num_cells, num_time_points);

decrease_activity = linspace(1, 0, num_time_points) ... 
    + normrnd(0, 1, num_cells, num_time_points);


baseline_noise_matrix = normrnd(0, 1, ... 
    num_cells, num_time_points, num_trials);

increase_activity_matrix = increase_activity + baseline_noise_matrix;
decrease_activity_matrix = decrease_activity + baseline_noise_matrix;

combined_matrices = cat(3, baseline_noise_matrix, ...
    increase_activity_matrix, decrease_activity_matrix);


% Optional: include string list of the dimension names
dim_names = ['Cell', 'Time', 'Trial'];

% Optional: include the coordinates associated with each dimension 
dim_struct.Cell.coord = 1:num_cells;
dim_struct.Time.coord = exp_time_coord;
dim_struct.Trial.coord = size(combined_matrices, 3); 


% Save the data
