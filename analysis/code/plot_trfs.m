clearvars -except user_profile boxdir_mine boxdir_lab
%note: plotting sep conditions gives errors when constructing avg models
%AND ind models havve 3 cols even though should be 1...

%% Script overview
% This script loads previously fit mTRF models for a set of subjects and
% experimental conditions, optionally filters channels based on model
% performance (correlation r), and visualizes:
%   - Individual-subject TRF weights
%   - Subject-averaged TRF weights
%   - Scalp topographies at selected latencies
%   - (Optional) SNR estimates
%   - (Optional) TRF peak latency analyses across electrodes and conditions
%
% The script assumes:
%   - A consistent folder structure and checkpoint format for TRF models
%   - That all subjects share compatible preprocessing and TRF configs
%   - Models were fit using the mTRF toolbox
%
% This is primarily a visualization / exploratory analysis script; it does
% not refit models.

%% Key configuration parameters (affect final output)
%
% script_config.experiment
%   Selects which experiment to analyze ('fast-slow' or 'reg-irreg').
%   This determines which subject IDs and condition labels are used.
%
% subjs
%   Explicit list of subject IDs to include. Automatically populated based
%   on experiment unless overridden.
%
% mtrf_plot_chns
%   Channels to plot in mTRFplot. Can be:
%     - 'all' (default)
%     - A numeric vector of channel indices
%
% script_config.show_individual_weights
%   If true, plots TRF weights for each subject × condition separately.
%
% script_config.show_avg_weights
%   If true, constructs and plots subject-averaged TRF models.
%
% script_config.show_topos
%   If true, plots scalp topographies of TRF weights at selected latencies.
%
% script_config.show_snr
%   If true, estimates and plots SNR based on RMS signal vs. noise windows.
%
% script_config.analyze_pk_latencies
%   If true, performs peak-finding on averaged TRFs to analyze component
%   latencies across electrodes and conditions.
%
% trf_fig_param
%   Struct controlling all TRF plotting behavior:
%     - t_lims        : x-axis (ms)
%     - ylims         : y-axis amplitude limits
%     - lw            : line width
%     - fz / title_fz : font sizes
%     - condition_colors : RGB color mapping by condition name
%     - r_thresh      : correlation threshold for channel inclusion
%     - stack         : whether to overlay conditions on a single axis
%     - pos           : figure size (inches)
%
% trf_fig_param.r_thresh
%   If non-empty, channels are retained only if their mean correlation
%   (across subjects) exceeds this threshold. This affects:
%     - Which channels contribute to subject-averaged plots
%     - Which channels are passed to mTRFplot
%
% topo_config.latencies
%   Latencies (ms) at which scalp topographies are plotted.

% Data flow summary
%
% 1. Loop over subjects:
%    - Load saved TRF model checkpoints
%    - Store models and configs
%    - (Optionally) extract correlation values for channel filtering
%
% 2. Optionally filter channels using r_thresh
%
% 3. Plot:
%    - Individual-subject TRFs (optional)
%    - Subject-averaged TRFs
%    - Stacked or separated condition plots
%
% 4. Optional analyses:
%    - SNR estimation
%    - Scalp topographies
%    - TRF peak latency detection and visualization
%
% 5. Helper functions at end handle:
%    - Averaging models across subjects
%    - SNR estimation
%    - Legend construction

%%TODO / known issues
%
% - Improve PSD/SNR analyses (currently simplistic RMS-based)
% - Make model-loading logic a standalone function
% - Consider generating TRF topo movies across time instead of fixed latencies
% - Resolve plotting issues when conditions are plotted separately
% - Remove assumptions that all configs except subject ID are identical
% TRF plot
%% plotting params
addpath('./trf_plot_lib/')
close all
script_config=[];

script_config.experiment='fast-slow';
script_config.custom_subjs=[];
script_config.topos_from_peak_windows=false;

% best fast-slow subjs: 
% subjs=[9,12];
if isempty(script_config.custom_subjs)
    switch script_config.experiment
        case 'fast-slow'
            % fast-slow subjs:
            subjs=[2:7,9:22];
            % subjs=22;
            butterfly_fig.ylims=[-.5 .6];
        case 'reg-irreg'
            % reg-irreg subjects:
            subjs=[23,96,97,98];
            butterfly_fig.ylims=[-1 1];
            % subjs=[96:98];
        otherwise
            warning('script_config.experiment=',script_config.experiment);
            disp('plotting custom subjs...')
    end
else
    % script_config.experiment=[];
    subjs=script_config.custom_subjs;
    butterfly_fig.ylims=[-1 1];
end

% for reg-irreg:
select_plot_chns='all'; % 'all' or vector with indices

mtrf_plot_chns=normalize_channels(select_plot_chns);
n_subjs=numel(subjs);
script_config.show_individual_weights=false;
script_config.show_avg_weights=true;
script_config.show_topos=false; % for particular latency specified in topo_fig_param
script_config.show_snr=false;
script_config.show_tuning_curves=false;
script_config.analyze_pk_latencies=false;
% params for select latency topos
topo_fig_param.latencies=[54 164]; % in ms
% params for butterfly with all conditions
butterfly_fig.t_lims=[-100 400];


butterfly_fig.lw=2; %linewidth in plot
butterfly_fig.leg_lw=6; % linewidth in legend
butterfly_fig.fz=11; % fontsize
butterfly_fig.title_fz=12;
% picked from Okabe and Ito pallete
butterfly_fig.condition_colors=struct( ...
    'slow',normalize([000 158 115],'norm'), ...
    'original',normalize([086 180 233],'norm'), ...
    'fast',normalize([230 159 000],'norm'), ...
    'reg',[255 0 0]./255, ...
    'irreg',[0 0 0]);
% r_thresh: correlation-based filter on TRF weights to show
% leave empty if all chns should be kept for sbj-averaged plot
% for fast slow
% trf_fig_param.r_thresh=[0.03];
% for reg-irreg
butterfly_fig.r_thresh=[];
% if false, separate TRF weight plots by condition
butterfly_fig.stack=true;
% x y width height - set to inches in figure
butterfly_fig.pos=[0 0 3 3];

% params for component plots (gpf + butterfly)
component_fig=butterfly_fig;




%% read data & setup grand average trfs

% preallocate
% note: all configs should be the same except for subj num and best_lam, so
% we probably don't need to keep all these in one large super structure
% but lets do that for now cuz we lazy and we might need them for
% reference later

configs=cell2struct(cell(n_subjs,2),{'preprocess_config','trf_config'},2);
 
for ss=1:n_subjs
    subj=subjs(ss);
    fprintf('loading subj %d data...\n',subj)

    trf_analysis_params;
    clear do_nulltest 
    % TODO: make function
    S_=load_checkpoint(trf_config);
    model_=S_.model;
    if ~isempty(butterfly_fig.r_thresh)
        rs_=squeeze(mean([S_.stats_obs(:).r],1));
        % gives conditions-by-chns, unless conditions missing
        if size(rs_,1)~=length(trf_config.conditions)
            % expand rs_ to proper size first... actually not sure what
            % exactly to do here.. issue is that with only a sinngle
            % condition rs_ becomes a col vector so conditions axis is
            % flipped... maybe can just transpose it
            if size(rs_,1)==length(rs_)
                rs_=rs_';
            end
            rs_=repmat(rs_,[3,1]);
            missing_cond=find(cellfun(@(x) isempty(x),{S_.stats_obs.r}));
            for ci=missing_cond
                rs_(ci,:)=nan;
            end
        end
    end
    clear S_
    if ss==1
        %preallocate
        model_fields_=fieldnames(model_);
        sz_=[size(model_),size(model_fields_)];
        sz_(1)=n_subjs;
        ind_models=cell2struct(cell(sz_),model_fields_,3);
        if ~isempty(butterfly_fig.r_thresh)
            sz_=[n_subjs, size(rs_)];
            rs=nan(sz_);
        end
        clear model_fields_ sz_
    end
    ind_models(ss,:)=model_; clear model_
    if ~isempty(butterfly_fig.r_thresh)
        rs(ss,:,:)=rs_; clear rs_
    end
    % function ends here
    
    configs(ss).preprocess_config=preprocess_config;
    configs(ss).trf_config=trf_config;
    clear preprocess_config trf_config
end
experiment_conditions=configs(end).trf_config.conditions;

% determine which channels to keep based on r_thresh
if ~isempty(butterfly_fig.r_thresh)
    fprintf('filtering chns based on r_thresh=%0.3f...\n',butterfly_fig.r_thresh)   
    % average out subjects & apply r_threshold
    chns_m=squeeze(mean(rs,1))>butterfly_fig.r_thresh;
    disp('chns remaining:')
    disp(sum(chns_m,2));
    if any(sum(chns_m,2)==0)
        warning('empty chns array will cause mtrfplot to show all the channels')
    end
end
% Note: loc_file defined in trf_analysis params...
load(loc_file);
avg_models=construct_avg_models(ind_models);

%% compute GFP
% calculates GFP (using standard deviation and not RMS because
% mastoid-referenced)
% w has size: [1 x time x channels]
gfp_grand=compute_gfp(avg_models, experiment_conditions);
% gfp_grand=nan(numel(experiment_conditions),size(avg_models(1).w,2));
% for cc = 1:numel(experiment_conditions)
%     W = squeeze(avg_models(1,cc).w);
%     gfp_grand(cc,:)=std(W,0,2);
%     % gfp(cc, :) = rms(W,2);
%     clear W
% end
% plot per-condition GFP:
gfp_plots=cell(size(experiment_conditions));
for cc=1:numel(experiment_conditions)
    gfp_plots{cc}=plot_gfp(gfp_grand,avg_models,cc,experiment_conditions,butterfly_fig);
end

%% estimate components via microstate analysis OR 
% basic gfp-peak picking algorithm

script_config.run_microstates=false;
script_config.run_basic_components=true;
if script_config.run_microstates
    get_microstates
end
if script_config.run_basic_components
    get_basic_components
end

% route whichever component source was run to a neutral variable
% plotting block below on reads active_components -- it never touches
% components or basic components directly
if script_config.run_microstates
    active_components=components;
    % components.result is a scalar struct (condition-pooled boundaries)
    get_active_result = @(cc) active_components.result;
elseif script_config.run_basic_components
    active_components=basic_components;
    % basic_components.result is a per-condition struct array
    get_active_result=@(cc) active_components.result(cc);
end
%% stacked butterfly + GFP plots with component boundaries
% as in Lalor et al 2009 Fig 4
switch script_config.experiment
    case 'fast-slow'
        gfp_ylim=[0 .2];
        trf_ylim= [-.6 .6];
    case 'reg-irreg'
        gfp_ylim=[0 .6];
        trf_ylim= [-.9 .9];
end
baseline_color=[.85 .85 .85];
for cc=1:numel(experiment_conditions)
    h = plot_butterfly_gfp(avg_models(cc), gfp_grand(cc,:), get_active_result(cc), ...
                                 experiment_conditions{cc}, butterfly_fig, ...
                                 gfp_ylim, trf_ylim, baseline(cc));
    plot_component_topos(get_active_result(cc), chanlocs, experiment_conditions{cc});
    % figure('Units','inches','Position',[0 0 7 3],'Name',...
    % sprintf('GFP with components plot %s', experiment_conditions{cc}), ...
    % 'Color','w');
    % 
    % % Define normalized axis positions
    % ax_gfp_pos = [0.12 0.70 0.83 0.22];   
    % ax_trf_pos = [0.12 0.12 0.83 0.58];
    % 
    % % --- GFP axis ---
    % ax_gfp = axes('Position', ax_gfp_pos);
    % plot(avg_models(cc).t,gfp_grand(cc,:), ...
    %     'Color',butterfly_fig.condition_colors.(experiment_conditions{cc}))
    % title(experiment_conditions{cc})
    % grid on
    % hold(ax_gfp,"on")
    % 
    % % add baseline indicator
    % plot(ax_gfp,butterfly_fig.t_lims,[baseline(cc), baseline(cc)],'--', ...
    %     'Color',baseline_color,'LineWidth',1);
    % hold(ax_gfp,"off")
    % ylabel('GFP')
    % set(ax_gfp, 'XTickLabel', [],'YLim',gfp_ylim)   % no x labels on top
    % box off
    % 
    % % --- Butterfly axis ---
    % ax_trf = axes('Position', ax_trf_pos);
    % h_=plot(avg_models(cc).t, squeeze(avg_models(cc).w));   % butterfly
    % grid on
    % % make them all the same color within a conditon
    % set(h_,'LineWidth',butterfly_fig.lw, ...
    %             'Color',butterfly_fig.condition_colors.(experiment_conditions{cc}))
    % set(ax_trf, 'YLim',trf_ylim)
    % ylabel('Amplitude')
    % xlabel('Time (ms)')
    % box off
    % 
    % % --- Annotate Component Windows ---
    % for kk=1:length(get_active_result(cc).starts)
    %     t1=get_active_result(cc).starts(kk);
    %     t2=get_active_result(cc).ends(kk);
    %     T=repmat([t1,t2],2,1);
    %     % put line in gfp plot
    %     hold(ax_gfp,"on");
    %     plot(ax_gfp,T, gfp_ylim, '--k')
    %     hold(ax_gfp,"off");
    %     % put line in trf plot
    %     hold(ax_trf,"on");
    %     plot(ax_trf,T, trf_ylim, '--k')
    %     hold(ax_trf,"off");
    %     clear t1 t2
    % end
    % % --- Synchronize ---
    % linkaxes([ax_gfp, ax_trf], 'x')
    % xlim(butterfly_fig.t_lims)

    % --- plot topos for each component ---
    for kk=1:size(get_active_result(cc).topos,1)
        figure('Units','inches','Position',[0 0 1 1.2], ...
            'Name',sprintf('component topo %s %0.0fms-0.0fms',experiment_conditions{cc}))
        topoplot(get_active_result(cc).topos(kk,:),chanlocs)
        title(sprintf('%s, %0.0fms-%0.0fms',experiment_conditions{cc}, ...
            get_active_result(cc).starts(kk),get_active_result(cc).ends(kk)))
        ax = gca;
        ax.LooseInset = max(ax.TightInset, 0.02);
    end
end

%% plot weights for individual subject
%TODO: abstract this into a function
if script_config.show_individual_weights
    for ss=1:n_subjs
        for cc=1:numel(configs(ss).trf_config.conditions)
            model_=ind_models(ss,cc);
            if ~isempty(model_.w)
                title_str_=sprintf('subj: %d - chns %s - %s',configs(ss).trf_config.subj, ...
                        num2str(mtrf_plot_chns),configs(ss).trf_config.conditions{cc});
                % plot_model_weights(ind_models(ss,:),trf_config,plot_chns)
                figure
                %NOTE: could filter chns here too but not a priority atm -
                %should do so on individual subject basis though probably?
                % if isempty(trf_fig_param.r_thresh)
                h_=mTRFplot(model_,'trf','all',mtrf_plot_chns);
 
                title(title_str_)
                title(title_str_,'FontSize',butterfly_fig.fz)
                set(h_,'LineWidth',butterfly_fig.lw)
            else
                fprintf('nothing to plot for %s\n', configs(ss).trf_config.conditions{cc})
            end 
            clear title_str_ h_ model_
        end
    end
end
%% Plot Grand average butterly 
if script_config.show_avg_weights
    for cc=1:numel(configs(end).trf_config.conditions)
        model_=avg_models(1,cc);
        if ~isempty(model_.w)

            if cc>1&&butterfly_fig.stack
                % skip adding fig
            else
                figure('Units','inches','Position',butterfly_fig.pos, ...
                    'Color','White','Name','Grand-Average Butterfly solo')
            end

            if isempty(butterfly_fig.r_thresh)
                h_=mTRFplot(model_,'trf','all',mtrf_plot_chns);
            else
                h_=mTRFplot(model_,'trf','all',find(chns_m(cc,:)));
            end
            % make all the lines within a condition the same color
            set(h_,'LineWidth',butterfly_fig.lw, ...
                'Color',butterfly_fig.condition_colors.(experiment_conditions{cc}))

            title_str='Grand-Average TRFs';
            title(title_str,'FontSize',butterfly_fig.title_fz)
            xlim(butterfly_fig.t_lims)
            ylim(butterfly_fig.ylims)
            ylabel('Amplitude (a.u.)')
            grid on
            set(gca,'FontSize',butterfly_fig.fz)
            if butterfly_fig.stack
                hold on
            else
                legend(experiment_conditions{cc})
            end
        else
            % useful for pilot subjects that only listen to a single
            % condition...
            fprintf('nothing to plot for %s\n',configs(ss).trf_config.conditions{cc})
        end
    end
    if butterfly_fig.stack
        hold off
        legend_helper(gca,experiment_conditions,butterfly_fig.condition_colors);
    end
    clear  h_ lg_
end
%% make and RGB-coded topoplot
script_config.show_rgb_topo=false;  
if script_config.show_rgb_topo
    % map x,y,z -> r,g,b
    X = [chanlocs.X];
    Y = [chanlocs.Y];
    Z = [chanlocs.Z];
    
    % Normalize to [0,1]
    R = (X - min(X)) ./ (max(X) - min(X));
    G = (Y - min(Y)) ./ (max(Y) - min(Y));
    B = (Z - min(Z)) ./ (max(Z) - min(Z));
    
    RGB = [R(:), G(:), B(:)];   % 128×3
    % N electrodes, M colormap entries
    % Result: N×M distance matrix
    D = squeeze(sum((RGB- permute(cmap,[3 2 1])).^2, 2));
    [~, idx] = min(D, [], 2);   % idx is N×1
    
    
    % get electrode x/y projections on scalp
    % ex = [chanlocs.radius] .* cos([chanlocs.theta]);
    % ey = [chanlocs.radius] .* sin([chanlocs.theta]);
    
    % draw head outline
    figure; hold on
    topoplot(idx, chanlocs, 'maplimits',[1 size(cmap,1)]);
    colormap(cmap);
    title("XYZ->RGB")
end

% REFERENCES
% Lalor, Edmund C., Alan J. Power, Richard B. Reilly, and John J. Foxe. 
%   "Resolving Precise Temporal Processing Properties of the Auditory System 
%   Using Continuous Stimuli.” Journal of Neurophysiology 102, no. 1 (2009): 
%   349–59. https://doi.org/10.1152/jn.90896.2008.
