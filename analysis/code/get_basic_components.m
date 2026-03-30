% BASIC COMPONENTS

%% time-based TRF component identification
% assumes GFP computed in plot_trfs
% For each condition:
%   - collapse across electrodes
%   - retain temporal structure

%% Identify candidate component latencies

basic_component_analysis.keep_top_n=[]; % leave empty if keeping all
% plot pre-smoothed GFP:
gfp_plots=cell(size(experiment_conditions));
for cc=1:numel(experiment_conditions)
    gfp_plots{cc}=plot_gfp(gfp,avg_models,cc,experiment_conditions,butterfly_fig);
end


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
    baseline(cc)=2*mean(gfp(cc,baseline_window_m),2);
    % Find local maxima above threshold
    % TODO: enforce minimum separation between peaks (did lalor et al do this?)
    % also, what would be a principled way to set that minimum separation value?
    [component_amplitudes,component_idx{cc}]=findpeaks(gfp(cc,:), ...
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
disp('Windowing around GFP peaks done.')
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
disp('GFP peaks identified and fixed-window topo average around them plotted.')