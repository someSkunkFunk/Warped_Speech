% optimize_sl_reset
% script for optimizing SL model on fast-slow data
% in order to fairly compare predicted TRFs, model should be fit
% independently for each subject & electrode... 

% --- IGNORE (NUISANCE SETUP) ---
 if ~(exist('EEG','var')&&exist('ENV','var')&&exist('PKRT','var')&& ...
         exist('COND','var'))
    % get EEG, ENV, & PKRT variables, each subj-by-trials cell
    load_all_fast_slow
    
 end
 clearvars -except boxdir_mine boxdir_lab EEG ENV PKRT COND
[n_subj,n_trials]=size(EEG);
fs=128;
n_electrodes=128;
% --- ---
%%
cond_labels={'fast','og','slow'};
sl=struct('param',[],'result',[]);
sl.out_path=fullfile(boxdir_mine,'analysis','sl_reset','optimized_models_fastSlow.mat');
% note: assuming small amplitude perturbations, and linearizing around 
% r_dot=0 lambda & gamma are somewhat redundant as different combinations
% r_stable=sqrt(lambda/gamma) the same way, namely the amplitude "relaxes"
% back to r_stable at a rate fully determined by lambda

% -> visualize 2D grid for a fixed value of gamma, then repeat for
% different values of gamma and compare them


% --- GRIDSEARCH SETTINGS ---
% reset model doesn't care about radius, set it to 1 for convenience
sl.param.lambda=1;
sl.param.gamma=1;


% -- fitting parameters -- 
sl.param.max_omega=32*2*pi; % rad/sec
sl.param.n_refinements=1; % number of grid refinements

% G1 is omega grid
% G2 is theta grid



sl.param.m_gridpoints=[10, 10];
% lambda, theta, INVERESE loss 
% (because maximizing in gridsearch -- check if this is valid thourgh)
sl.param.init_pts=ones(1,3).*(-1e-10); % should be small

% construct initial grid
G1=linspace(0,sl.param.max_omega, ...
    sl.param.m_gridpoints(1));
G2=linspace(0,2*pi,sl.param.m_gridpoints(2));

% preallocate array to store loss-function for all samples
% note: when grid is refined, can extend this by making it a cell and
% adding new inv_loss array to it
inv_loss=nan(n_subj,n_electrodes,numel(G1),numel(G2));
% --- ---

%%
% check if results have been precomputed and load, or run optimization
% analysis....
disp('sl reset params:')
disp(sl.param)
if exist(sl.out_path,'file')==0||sl.overwrite
   
    % initialize values
    optimal.omega=sl.param.init_pts(1).*ones(size(EEG));
    optimal.theta=sl.param.init_pts(2).*ones(size(EEG));
    optimal.inv_loss=sl.param.init_pts(3).*ones(size(EEG));
    % Build sim_param struct array (one entry per condition)
    % sim_param = struct('dt', [], 'T', [], 't', [], 'N', []);
    % for cc = 1:length(cond_labels)
    %     sim_param(cc).dt = 1 / fs;
    %     sim_param(cc).T  = (length(ENV{cc,1}) - 1) / fs;
    %     sim_param(cc).t  = 0 : sim_param(cc).dt : sim_param(cc).T;
    %     sim_param(cc).N  = length(sim_param(cc).t);
    % end
 
    % fprintf('GridSearch %d of %d (subj %d, trial %d)...\n',pt, ...
    %     numel(EEG),pp,tt)
    for rr=1:sl.param.n_refinements
        for ff=1:length(G1)
            for aa=1:length(G2)
                %% Simulate SL oscillator for every participant x stimulus
                for pp=1:size(EEG,1)
                    x_sl_={};
                    for tt=1:size(EEG,2)
                        % Build sim_param struct array (note: there's ultimately only 3 
                        % possible outcomes for this but it depends on the trial order)
                        sim_param = struct('dt', [], 'T', [], 't', [], 'N', []);
                        sim_param.dt = 1 / fs;
                        sim_param.T  = (length(ENV{pp,tt}) - 1) / fs;
                        sim_param.t  = 0 : sim_param.dt : sim_param.T;
                        sim_param.N  = length(sim_param.t);
                        sl_=sl;
                        sl_.param.f_nat=G1(ff)/(2*pi); % sl_reset assumes Hz for frequency
                        sl_.param.theta_r=G2(aa);
                        % simulated response
                        x_sl_={x_sl_,sl_reset(sl_, PKRT{pp,tt}.pkTimes, sim_param)};
                    end
                    % calculate RMSE over all responses for current
                    % subject and reset temp x_sl var
                    % inv_loss(pp,:,ff,aa)=sqrt(mean(EEG{pp,}))
                end
                   

                


                if inv_loss(tt,ff,aa)>optimal.inv_loss(tt)
                    optimal.inv_loss(tt)=inv_loss(tt,ff,aa);
                    optimal.omega(tt)=G1(ff);
                    optimal.theta(tt)=G2(aa);
                end
            end
        end
    end

    disp('GridSearch complete')
    sl.result.inv_loss=inv_loss;
    sl.result.optimal=optimal;
    fprintf('Saving results to %s...\n',sl.out_path)

else
    load(sl.out_path);
end
%%
function x = sl_reset(sl, event_times, sim_param)
% Simulate Stuart-Landau oscillator with hard phase resets.
    omega = 2 * pi * sl.param.f_nat;

    dt = sim_param.dt;
    t  = sim_param.t;
    N  = sim_param.N;

    r0     = sqrt(sl.param.lambda / sl.param.gamma);
    theta0 = 2*pi*rand(1);
    r     = zeros(1, N);
    theta = zeros(1, N);
    r(1)     = r0;
    theta(1) = theta0;

    event_times = sort(event_times(:)');
    event_idx   = 1;

    f_r = @(ri) sl.param.lambda * ri - sl.param.gamma * ri^3;

    for i = 1:N-1
        % Hard phase reset if this sample coincides with an event
        if event_idx <= length(event_times) && ...
                abs(t(i) - event_times(event_idx)) < dt/2
            theta(i)  = sl.param.theta_r;
            event_idx = event_idx + 1;
        end

        % RK4 for r
        ri = r(i);
        k1 = f_r(ri);
        k2 = f_r(ri + dt/2 * k1);
        k3 = f_r(ri + dt/2 * k2);
        k4 = f_r(ri + dt   * k3);
        r(i+1)     = ri + (dt/6) * (k1 + 2*k2 + 2*k3 + k4);
        theta(i+1) = theta(i) + omega * dt;
    end

    % Final sample reset check
    if event_idx <= length(event_times) && ...
            abs(t(N) - event_times(event_idx)) < dt/2
        theta(N) = sl.param.theta_r;
    end

    x = r .* cos(theta);
end

