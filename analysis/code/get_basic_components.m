% BASIC COMPONENTS
% Output: basic_components struct with fields:
%   basic_components.result.starts - component window start times (ms),
%       [n_components x1]
%   basic_components.result.ends - component window end times (ms),
%       [n_components x 1]
%   basic_components.result.topos - mean topography per window,
%       [n_components x n_channels]
% Schema intentionally matches components.result from get_microstates.m so
% either source can be passed to the butterfly+GFP plotting block in
% plot_trfs.m

% time-based TRF component identification
% assumes GFP computed in plot_trfs
% For each condition:
%   - collapse across electrodes
%   - retain temporal structure

%% Identify candidate component latencies

basic_component_analysis.keep_top_n=2; % leave empty if keeping all
% limit number of peaks to consider a component based on amplitude...
component_idx=cell(size(experiment_conditions));
component_times=cell(size(experiment_conditions));
baseline=zeros(3,1);
% limit search range because there are edge artefacts.... but how do we
% decide objectively what time range makes sense???? TODO!
basic_component_analysis.tbounds=[0, 500];
for cc = 1:numel(experiment_conditions)


    % Define an objective threshold
    % Using same as Lalor et al. (2009) - twice the mean GFP during
    % -100ms,0ms window
    baseline_window_m=avg_models(cc).t<0 & avg_models(cc).t>-100;
    baseline(cc)=2*mean(gfp_grand(cc,baseline_window_m),2);
    % Find local maxima above threshold
    % TODO: enforce minimum separation between peaks (did lalor et al do this?)
    % also, what would be a principled way to set that minimum separation value?
    [component_amplitudes,component_idx{cc}]=findpeaks(gfp_grand(cc,:), ...
        "MinPeakHeight",baseline(cc)+eps);


    if ~isempty(basic_component_analysis.keep_top_n)
        [~,sortI]=maxk(component_amplitudes,basic_component_analysis.keep_top_n);
        component_idx{cc}=component_idx{cc}(sortI);
    end
    % filter peaks so only looking in search window defined by tbounds
    component_times{cc}=avg_models(cc).t(component_idx{cc});

    tbounds_m_=component_times{cc}<max(basic_component_analysis.tbounds)&...
        component_times{cc}>min(basic_component_analysis.tbounds);
    component_idx{cc}=component_idx{cc}(tbounds_m_);
    component_times{cc}=component_times{cc}(tbounds_m_);
    clear tbounds_m_
end

%% define component latency windows
% component start,end in ms relative to component time
basic_component_analysis.component_window_ms=[-16 16];
% doing this somewhat arbitrarily based on the minimum size of windows
% reported in Lalor et al. 2009 - but it would be cool to use the actual
% microstates analysis they used to determine a better component...

% apparently 10 ms window (the "minimum" size referred to above) is too 
% small since we sampled at 128 Hz our delta is ~7.8 ms so bumping it up to
% the average size given by window lims reported in Lalor et al 2009:
% mean(diff([45,  61, 92, 104, 125, 170, 238])) ~32


component_windows=cell(size(experiment_conditions));
for cc = 1:numel(experiment_conditions)
    for kk = 1:numel(component_idx{cc})
        t_range_ms_=component_times{cc}(kk)+basic_component_analysis.component_window_ms;
        t_start_idx_=find(avg_models(cc).t>min(t_range_ms_),1,'first');
        t_end_idx_=find(avg_models(cc).t<max(t_range_ms_),1,'last');
        component_windows{cc}(kk,:) = [t_start_idx_, t_end_idx_];
        clear t_range_ms_ t_start_idx_ t_end_idx_
    end
end
disp('Windowing around gfp_grand peaks done.')
% extract and plot component topographies
component_topos=cell(size(experiment_conditions));
for cc=1:numel(experiment_conditions)
    for kk=1:numel(component_idx{cc})
        win=component_windows{cc}(kk,1):component_windows{cc}(kk,2);
        % average out the weights (TODO: do we wanna look at pre-averaged
        % topos too?)
        component_topos{cc}(kk,:)=squeeze(mean(avg_models(cc).w(1,win,:),2));
    end
end
% plotting topos
for cc=1:numel(experiment_conditions)
    for kk=1:numel(component_idx{cc})
        figure
        topoplot(component_topos{cc}(kk,:),chanlocs)
        title(sprintf('%s - %.0f ms',experiment_conditions{cc},component_times{cc}(kk)))
    end
end
disp('gfp_grand peaks identified and fixed-window topo average around them plotted.')

%% pack results into shared schema
basic_components=struct('param',[],'result',[]);
basic_components.param.method = 'grp_peaks';
basic_components.param.tbounds=basic_component_analysis.tbounds;
basic_components.param.window_ms=basic_component_analysis.component_window_ms;

% collect starts/ends/topos across all conditions
% (one struct entry per condition, parallel to components from
% get_microstates)
for cc=1:numel(experiment_conditions)
    n_comp=numel(component_idx{cc});
    starts_=nan(n_comp,1);
    ends_=nan(n_comp,1);
    for kk=1:n_comp
        starts_(kk)=avg_models(cc).t(component_windows{cc}(kk,1));
        ends_(kk)=avg_models(cc).t(component_windows{cc}(kk, 2));
    end
    basic_components.result(cc).starts=starts_;
    basic_components.result(cc).ends=ends_;
    basic_components.result(cc).topos=component_topos{cc}; % [n_comp x n_chns]
    clear starts_ ends_ n_comp
end
disp('basic_components struct packed.')