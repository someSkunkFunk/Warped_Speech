% optimize_sl_reset
% script for optimizing SL model on fast-slow data
% in order to fairly compare predicted TRFs, model should be fit
% independently for each subject & electrode... 
clearvars -except boxdir_mine boxdir_lab
% --- IGNORE (NUISANCE SETUP) ---
load_all_fast_slow
% convert envelopes to peakrate (dumb, not oganian)

% --- ---
%%
% --- LOAD PREPROCESSED EEG ---

sl_reset_gridsearch=struct('param',[],'result',[]);
sl_reset_gridsearch.out_path=fullfile(boxdir_mine,'analysis','sl_reset','optimized_models_fastSlow.mat');
% note: assuming small amplitude perturbations, and linearizing around 
% r_dot=0 lambda & gamma are somewhat redundant as different combinations
% r_stable=sqrt(lambda/gamma) the same way, namely the amplitude "relaxes"
% back to r_stable at a rate fully determined by lambda

% -> visualize 2D grid for a fixed value of gamma, then repeat for
% different values of gamma and compare them


% --- GRIDSEARCH SETTINGS ---
% reset model doesn't care about radius, set it to 1 for convenience

sl_reset_gridsearch.param.lambda=1;
sl_reset_gridsearch.param.gamma=1;

sl_reset_gridsearch.param.n_refinements=1; % number of grid refinements

% G1 is lambda grid
% G2 is theta grid

%note: very small value was used previously, so extending the range might
%produce weird results... play with this parameter
sl_reset_gridsearch.param.lambda=1;
sl_reset_gridssearch.param.gamma=1;

% note lambda oscillatory regime is lambda>0

% note: I think it's safe to assume less fine-spaced grid along theta axis
% is sufficient
sl_reset_gridsearch.param.m_gridpoints=[10, 4];
% lambda, theta, INVERESE loss 
% (because maximizing in gridsearch -- check if this is valid thourgh)
sl_reset_gridsearch.param.init_pts=ones(1,3).*(-1e-10); % should be small

% construct initial grid
G1=linspace(0,sl_reset_gridsearch.param.max_lambda, ...
    sl_reset_gridsearch.param.m_gridpoints(1));
G2=linspace(0,2*pi,sl_reset_gridsearch.param.m_gridpoints(2));
% --- ---

%%
% check if results have been precomputed and load, or run optimization
% analysis....
if exist(sl_reset_gridsearch.out_path,'file')==0||sl_reset_gridsearch.overwrite
    % load fast-slow envelopes
    fasl_envs_path=fullfile(boxdir_mine,'stimuli','wrinkle','fastSlowEnvelopes128Hz.mat');
    % note: not sure if normalizing or anything like that is desireable here?
    load(fasl_envs_path,'env','fs');
    % load preprocessed eeg, one subject at a time


    % initialize values
    optimal.lambda=sl_reset_gridsearch.param.init_pts(1);
    optimal.theta=sl_reset_gridsearch.param.init_pts(2);
    optimal.inv_loss=sl_reset_gridsearch.param.init_pts(3);

    for rr=1:sl_reset_gridsearch.param.n_refinements
        for ll=1:length(G1)
            for tt=1:length(G2)
                %% Simulate SL oscillator for every condition x stimulus
                % x_all{cc,ss} = oscillator output vector for condition cc, stimulus ss
                x_all = cell(n_cond, n_stim);
                
                fprintf('Simulating SL oscillator (%d conditions x %d stimuli)...\n', n_cond, n_stim);
                for cc = 1:n_cond
                    for ss = 1:n_stim
                        x_all{cc,ss} = sl_reset(sl, event_times{cc,ss}, sim_param(cc));
                    end
                    fprintf('  Condition %d done.\n', cc);
                end

                %TODO: RMSE evaluation should be independent for each
                %subject, electrode....
                % inv_loss= % -(RMSE)

                if inv_loss>optimal.inv_loss
                    optimal.inv_loss=inv_loss;
                    optimal.lambda=G1(ll);
                    optimal.theta=G2(tt);
                end
            end
        end
    end

else
    load(sl_reset_optim.out_path);
end
%%
