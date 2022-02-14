function [dat, T, N, F, names, target_neuron_id, cumsum_explained] = GAM_Step1(file,target_neuron, binSize)

% inputs target_neuron, which is the i-th neuron in a specific .mat
% (session/area). Output a target_neuron_id, which is 
% dat.n(target_neuron).id = target_neuron_id

%% Params
window = [-.5 1.5];

params.startRange = window(1);
params.stopRange = window(2);
params.binSize = binSize; 
params.n_dim = 10; 

time = params.startRange:params.binSize:params.stopRange;

%%

%% Load
load(file);


%% Initialize
contrast_dic = [-1.0000, -0.2500, -0.1250, -0.0625, 0, 0.0625, 0.1250, 0.2500, 1.0000];
prior_dic = [.2, .5, .8];
choice_dic = [-1, 0, 1];
feedback_dic = [-1, 1];
movement_dic = [-1, 1];

trial_vector = [];
contrast_vector = [];
choice_vector = [];
choice_history_vector = [];
feedback_vector = [];
feedback_history_vector = [];
movement_vector = [];

exp_prior_vector = [];
sub_prior_vector = [];

score_vector = []; 

%% Behavior
for i_trial = 1:size(dat.b.t.stimOn, 1)
    
    % trial vectors
    trial_vector = [trial_vector, i_trial*ones(1,size(time, 2))];
    
    % contrast vector
    if  ~isnan(dat.b.contrastL(i_trial))
        temp_contrast =  - dat.b.contrastL(i_trial);
    else
        temp_contrast = dat.b.contrastR(i_trial);
    end
    idx = find(temp_contrast == contrast_dic);
    temp_contrast_vector = zeros(9, size(time, 2));
    temp_contrast_vector(idx, abs(params.startRange)/params.binSize) = 1;
    contrast_vector = horzcat(contrast_vector, temp_contrast_vector);
    clear temp_contrast_vector idx
    
    % choice_vector
    try
        col_idx = abs(params.startRange)/params.binSize + floor((dat.b.t.movement(i_trial) - dat.b.t.stimOn(i_trial))/params.binSize);
        row_idx = find(dat.b.choice(i_trial) == choice_dic);
        temp_choice_vector = zeros(3, size(time, 2));
        if col_idx < size(time, 2)
            temp_choice_vector(row_idx, col_idx) = 1;
        end
        choice_vector = horzcat(choice_vector, temp_choice_vector);
        clear temp_choice_vector col_idx row_idx
    catch
        choice_vector = horzcat(choice_vector, temp_choice_vector);
        clear temp_choice_vector col_idx row_idx
    end
    
    
    % choice_vector t-1
    if i_trial == 1
        temp_choice_vector = zeros(3, size(time, 2));
    else
        idx = find(dat.b.choice(i_trial-1) == choice_dic);
        temp_choice_vector = zeros(3, size(time, 2));
        temp_choice_vector(idx, 1:size(time, 2)) = 1;
    end
    choice_history_vector = horzcat(choice_history_vector, temp_choice_vector);
    clear temp_choice_vector idx
    
    % feedback
    try
        col_idx = abs(params.startRange)/params.binSize + floor((dat.b.t.feedback(i_trial) - dat.b.t.stimOn(i_trial))/params.binSize);
        row_idx = find(dat.b.feedback(i_trial) == feedback_dic);
        temp_feedback_vector = zeros(2, size(time, 2));
        if col_idx < size(time, 2)
            temp_feedback_vector(row_idx, col_idx) = 1;
        end
        %size_feedback = [size_feedback, size(temp_feedback_vector, 2)];
        feedback_vector = horzcat(feedback_vector, temp_feedback_vector);
        clear temp_feedback_vector col_idx row_idx
    catch
        temp_feedback_vector = zeros(2, size(time, 2));
        feedback_vector = horzcat(feedback_vector, temp_feedback_vector);
        clear temp_feedback_vector col_idx row_idx
    end
    
    % feedack history
    try
        if i_trial == 1
            temp_feedback_vector = zeros(2, size(time, 2));
        else
            col_idx = abs(params.startRange)/params.binSize + floor((dat.b.t.feedback(i_trial) - dat.b.t.stimOn(i_trial))/params.binSize);
            row_idx = find(dat.b.feedback(i_trial-1) == feedback_dic);
            temp_feedback_vector = zeros(2, size(time, 2));
            if col_idx < size(time, 2)
                temp_feedback_vector(row_idx, 1:size(time, 2)) = 1;
            end
        end
        feedback_history_vector = horzcat(feedback_history_vector, temp_feedback_vector);
        clear temp_feedback_vector col_idx row_idx
    catch
        temp_feedback_vector = zeros(2, size(time, 2));
        feedback_history_vector = horzcat(feedback_history_vector, temp_feedback_vector);
        clear temp_feedback_vector col_idx row_idx
    end
    
    % movement
    try
        col_idx = abs(params.startRange)/params.binSize + floor((dat.b.t.movement(i_trial) - dat.b.t.stimOn(i_trial))/params.binSize);
        row_idx = find(dat.b.choice(i_trial) == movement_dic);
        temp_movement_vector = zeros(2, size(time, 2));
        if col_idx < size(time, 2)
            temp_movement_vector(row_idx, col_idx) = 1;
        end
        movement_vector = horzcat(movement_vector, temp_movement_vector);
        clear temp_movement_vector col_idx row_idx
    catch
        temp_movement_vector = zeros(2, size(time, 2));
        movement_vector = horzcat(movement_vector, temp_movement_vector);
        clear temp_movement_vector col_idx row_idx
    end
        
    
    
    %%
    % experimenter prior
    idx = find(dat.b.probLeft(i_trial) == prior_dic);
    temp_prior_vector = zeros(3, size(time, 2));
    temp_prior_vector(idx, :) = 1;
    exp_prior_vector = horzcat(exp_prior_vector, temp_prior_vector);
    clear temp_prior_vector idx;
    
    % subjective prior
    sub_prior_vector = horzcat(sub_prior_vector, dat.b.estimated_prior(i_trial)*ones(1, size(time, 2)));
    
    
end


%% neurons
neuron_vector = [];

for i_neuron = 1:size(dat.n, 2)
    
    temp_neuron_vector = [];
    
    for i_trial = 1:size(dat.b.t.stimOn, 1)
        try
            t_pre = dat.b.t.stimOn(i_trial) + params.startRange;
            t_post = dat.b.t.stimOn(i_trial) + params.stopRange;
            
            ss = dat.n(i_neuron).st(find(dat.n(i_neuron).st > t_pre & dat.n(i_neuron).st < t_post)) - dat.b.t.stimOn(i_trial);
            ba = hist(ss,time);yt(1)=0;yt(end)=0;yt=yt(:);
        catch
            ba =  nan(1, size(time, 2));
        end
        
        temp_neuron_vector = horzcat(temp_neuron_vector, ba);
    end
    
    neuron_vector(i_neuron, :) =  temp_neuron_vector;
    neuron_id(i_neuron) = dat.n(i_neuron).id; 
    clear temp_neuron_vector
end



%% Movement

try
    data_camera = dat.c.right.d(:, [1, 2, 4, 5, 7, 8, 10 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26, 28, 29, 31, 32]);
    [coeff,score,latent,tsquared,explained,mu] = pca(zscore(data_camera));
    score_time = dat.c.right.t;

for i_trial = 1:size(dat.b.t.stimOn, 1)
    clear t_pre t_post temp_score
    try
        t_pre = dat.b.t.stimOn(i_trial) + params.startRange;
        t_post = dat.b.t.stimOn(i_trial) + params.stopRange;
        
        temp_score = score((score_time > t_pre & score_time < t_post), 1:params.n_dim)'; 
        
        % interpolating
        for i_int = 1:params.n_dim
            x = (1:size(temp_score, 2))';
            y = temp_score(i_int, :)';
            xi = linspace(1, size(temp_score, 2), size(time, 2)); 
            %xi = ( 1:(size(temp_score, 2)/403):size(temp_score, 2))';
            yi = interp1q(x,y,xi);
            temp_score_int(i_int, :) = yi';
        end
        score_vector = horzcat(score_vector, temp_score_int); 
        clear temp_score
    catch
        temp_score = nan(params.n_dim, size(time, 2)); 
        score_vector = horzcat(score_vector, temp_score);
        clear temp_score
    end
end
cumsum_explained = cumsum(explained);
cumsum_explained = cumsum_explained(1:params.n_dim); 
catch
    score_vector = nan(params.n_dim, size(contrast_vector, 2)); 
    cumsum_explained = nan(1, params.n_dim); 
end

%%

% t_move = find(sum(movement_vector)); 
% 
% figure; 
% plot(score_vector(1, 1:10000), 'k'); hold on;
% plot(score_vector(2, 1:10000), 'g'); hold on;
% plot(score_vector(3, 1:10000), 'm'); hold on;
% for i = 1:30
%     plot([t_move(i) t_move(i)], [-10 10], 'r'); hold on; 
% end


%% Putting it all together

T = trial_vector;
N = neuron_vector(target_neuron, :);
neuron_vector(target_neuron, :) = [];
target_neuron_id = neuron_id(target_neuron); 
neuron_id(target_neuron) = []; 
F = vertcat(contrast_vector, choice_vector, choice_history_vector, feedback_vector, feedback_history_vector, score_vector, exp_prior_vector, sub_prior_vector);
F = vertcat(F, neuron_vector);

names = {'cL100', 'cL025', 'cL012', 'cL006', 'c0000', 'cR006', 'cR012', 'cR025', 'cR100',...
    'choiceL', 'choice0', 'choiceR', ...
    'prev_choiceL', 'prev_choice0', 'prev_choiceR', ...
    'feedback_correct', 'feedback_incorrect', ...
    'prev_feedback_correct', 'prev_feedback_incorrect', ...
    'movement_PC1', 'movement_PC2', 'movement_PC3', 'movement_PC4', 'movement_PC5', 'movement_PC6', 'movement_PC7', 'movement_PC8', 'movement_PC9', 'movement_PC10',...
    'prior20','prior50', 'prior80',...
    'subjective_prior'};

counter = 0; 
for i = 34:size(F, 1)
    counter = counter + 1; 
    names{i} = ['neuron_' num2str(neuron_id(counter))];
end

end



