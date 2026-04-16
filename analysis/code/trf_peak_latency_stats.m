% plot_trfs -> trf_peak_latency
% computes a spatial average over electrodes (optionally confined to an ROI
% defined by peak prominence
%% -- GENERAL ANALYSIS SETTINGS ---

% copy component_windows so we can manipulate it independently of component
% analysis script result here
stats_windows=component_windows;
peaktime_stats={};
peaktime_stats.opts=struct('prominence_thresh',[], ...
    'match_component_windows',[],'drop_peaks',[]);

% roi_selection: reduce TRFs for more reliable statistics
% 'all': compute spatial average across all electrodes.
% 'prominent peaks': defines an ROI based on which electrodes have
% above-threshold peak prominence in grand average
peaktime_stats.opts.prominence_thresh=0;
% drop_peaks: ignore components whose bounds contain peak at latency
% specified (using because N1 not reliable in individual subject TRFs)
% do this operation before match_component_windows so extra components
% aren't added just to be removed again
% peaktime_stats.opts.drop_peaks=100; % in ms
% if ~isempty(peaktime_stats.opts.drop_peaks)
%     drop_peaks_=peaktime_stats.opts.drop_peaks;
%     drop_peaks_
%     for tt=1:length(drop_peaks_)
%         for cc=1:length(experiment_conditions)
%             t_=avg_models(cc).t;
%             for ci=size(component_windows{cc},1)
%                 win_=component_windows{cc}(ci,:);
%                 if drop_peaks_(tt)>t_(win_(1)) && drop_peaks_(tt)<t_(win_(2))
% 
%                 end
%             end
%         end
%     end
% end
% % remove outside of loop to avoid indexing errors from in-place removal
% clear drop_peaks_ t_ win_
% match_component_windows: empty,'max' or 'min', normalizes component 
% windows to be the same across conditions, assuming the same underlying 
% windows make sense across conditions (as in fast-slow)
% 'max': matches component windows in each condition to the condition with
% the maximum number of components
% 'min': matches component windows in each condition to the condition with
% the minimum number of components
peaktime_stats.opts.match_component_windows='min';
if ~isempty(peaktime_stats.opts.match_component_windows)
    % assumes component windows line up across conditions, otherwise this
    % doesn't make sense
    components_per_condition_=cellfun(@(x) size(x,1), stats_windows);
    switch peaktime_stats.opts.match_component_windows
        case 'max'
            N_=max(components_per_condition_);
        case 'min'
            N_=min(components_per_condition_);
    end
    template_windows_=stats_windows{find(components_per_condition_==N_,1)};
    for cc=1:length(stats_windows)
        if components_per_condition_(cc)~=N_
            stats_windows{cc}=template_windows_;
        end
    end
end
clear N_ components_per_condition_ template_windows_
%% -- select electrodes using TRF peak-prominence threshold --
% subsequent analyses then restricted to average subject-level TRFs across
% ROI defined by peak-prominence

% preallocate cell array to contain prominence-threshold masks
selected_electrodes=cellfun(@(x) repmat({false(1,128)},1,size(x,1)), ...
    stats_windows,'UniformOutput',false); %{1 x conditions}, {1 x components}, [1 x electrodes]
peaktimes_grand=cellfun(@(x) repmat(cell(1,128),size(x,1),1), ...
    stats_windows,'UniformOutput',false); %{1 x conditions}, {components x electrodes}
% -- plot topo with selected electrodes marked alongside color-coded
% butterfly 
for cc=1:numel(experiment_conditions)
    w_=avg_models(cc).w;
    % rectify weights before peak-picking
    w_rect_=sign(w_).*w_;
    for ci=1:size(stats_windows{cc},1)
        win_=stats_windows{cc}(ci,:); % indices, not time 
        % (relative to full lag window)
        t_win_=avg_models(cc).t(win_(1):win_(2));
        for ee=1:128
            [~,peaklocs_]=findpeaks(w_rect_(win_(1):win_(2)), ...
                'MinPeakProminence',peaktime_stats.opts.prominence_thresh);
            % TODO: find prominence threshold where a single peak maximum
            % per electrode... NOTE: with no threshold, it seems like all
            % electrode TRFs only contain a single peak in window
            % already...
            if ~isempty(peaklocs_)
                if length(peaklocs_)>1
                    warning(['subsequent code assumes each windowed ' ...
                        'grand-TRF contains a single peak maximum based on' ...
                        ' initial visual inspection.'])
                end
                % identify electrodes with at least one peak in windowed 
                % grand-trf -- store peaktimes for later
                peaktimes_grand{cc}{ci,ee}=t_win_(peaklocs_);
                selected_electrodes{cc}{ci}(ee)=true;
            end
        end
    end
end
clear peaklocs_ t_win_ win_ w_ w_rect_ ee


%% -- visualize electrode selection on grand-TRFs
% --- layout constraints ---
N_=numel(experiment_conditions);
M_=max(cellfun(@(x) size(x,1), stats_windows));
ncols_=M_*2; % each component window occupies two subplot columns
nrows_=N_;
topo_limits_=[]; % []= auto or some numeric values
if isempty(topo_limits_)
    topo_limits_='absmax';
end
% -------
titles_=cell(N_,M_);
for cc=1:N_
    for ci=1:size(stats_windows{cc},1)
        times_ms_=avg_models(cc).t; % [1 x time] in ms
        win_=stats_windows{cc}(ci,:);
        start_t_=times_ms_(win_(1));
        end_t_=times_ms_(win_(2));
        titles_{cc,ci}=sprintf('%s, %0.0f ms-%0.0f ms', ...
            experiment_conditions{cc},start_t_, ...
            end_t_);
    end
end
clear cc ci times_ms_ start_t_ end_t_
%
figure('Name', 'selected electrodes for subject-level peak latency analysis', ...
    'Units', 'inches','Position',[1 1 8 6], ...
    'Color','w')
t=tiledlayout(nrows_,ncols_,'TileSpacing','tight','Padding','tight');
for cc=1:N_
    for ci=1:M_
        % note : since moving across tiled layout using nexttile, if
        % condition,window pair is not in original component_windows, we
        % should skip this tile
        if ci<=numel(selected_electrodes{cc})       
            % get mask for electrodes containing a prominent peak
            prominent_elecs_mask_=selected_electrodes{cc}{ci}(:);
            trf_=squeeze(avg_models(cc).w)'; % [chn x time]
            times_ms_=avg_models(cc).t; % [1 x time, ms]
            ttl_=titles_{cc,ci};
    
            % --- get column indices in the tiled grid ---
            col_butterfly=(ci-1)*2+1;
            col_topo=(ci-1)*2+2;
            tile_idx_but=(cc-1)*ncols_+col_butterfly;
            tile_idx_topo=(cc-1)*ncols_+col_topo;
            
            % --- Butterfly plot ---
            nexttile(tile_idx_but);
            
            % Note: check plot below has correct dimensions
            plot(times_ms_, trf_(prominent_elecs_mask_,:), ...
                'Color', [0.6 0.6 0.6 0.4],'LineWidth',0.8)
            hold on;grid on
            plot(times_ms_, mean(trf_(prominent_elecs_mask_,:),1),'k','LineWidth',2);
        
            % shade topoplot window
            ylims_=ylim;
            win_=stats_windows{cc}(ci,:);
            toi_ms_=[times_ms_(win_(1)),times_ms_(win_(2))];
            patch([toi_ms_(1) toi_ms_(2) toi_ms_(2) toi_ms_(1)], ...
                [ylims_(1) ylims_(1) ylims_(2) ylims_(2)], ...
                [1 0.8 0.2], 'FaceAlpha',0.15,'EdgeColor','none', ...
                'HandleVisibility','off');
    
            xlabel('Time (ms)'); ylabel('Amplitude (a.u.)');
            title(ttl_);
            xlim([-100 400]);
            box off;
    
            % --- Topoplot ---
            nexttile(tile_idx_topo);
            toi_mask_=times_ms_>=toi_ms_(1) & times_ms_<=toi_ms_(2);
            % average over component time window
            % mark electrodes included in peak-latency analysis average TRF
            % using emarker
            topo_data_=mean(trf_(:,toi_mask_),2);
            topoplot(topo_data_, chanlocs, ...
                'maplimits', topo_limits_, ...
                'electrodes', 'on', ...
                'emarker2', {find(prominent_elecs_mask_), 'x', 'k', 1, .1})
            colorbar;
            axis off;
        end
    end
end

clear N_ M_ titles_ ncols_ nrows_ cmap_ titles_ prominent_elecs_mask_ ttl_ times_ms_ ylims_ win_ toi_ms_ topo_data_ toi_mask_ topo_data_
title(t, sprintf('min prominence: %0.2f',peaktime_stats.opts.prominence_thresh), 'FontSize',14)
%% --- identify peaks in individual subject TRFs---
% for each subject/electrode within predefined component windows


peaktimes_subjlvl=cellfun(@(x) repmat({nan(n_subjs,1)},1,size(x,1)), ...
    stats_windows,'UniformOutput',false); % {1 x conditions}, {1 x components}, [subjs x 1]

for ss=1:n_subjs
    for cc=1:numel(experiment_conditions)
        w_=ind_models(ss,cc).w;
        t_=ind_models(ss,cc).t;

        w_avg_=squeeze(mean(w_,3));
        w_avg_rect_=sign(w_avg_).*w_avg_;


        for ci = 1:size(stats_windows{cc},1)
            win_=stats_windows{cc}(ci,:); % indices, not time
            % [peakvals_,peaklocs_]=findpeaks(w_avg_rect_(win_(1):win_(2)));
            % some TRFs just dont peak in expected windows based on grand
            % averages
            [peakvals_,peaklocs_]=max(w_avg_rect_(win_(1):win_(2)));
            % note: i think this might be overkill with spatially-averaged
            % TRFs but maybe not -- sort peaks so max-amplitude peak is
            % selected
            % if there are multiple peaks within window, it might be worth
            % using a geometric mean or something to get peak lag
            if length(peaklocs_)>1
                [~,idx_]=sort(peakvals_);
                peaklocs_sorted_=peaklocs_(idx_);
            else
                peaklocs_sorted_=peaklocs_;
            end
            % trim trf time vector to component window lags so peaklocs
            % matches correct time
            t_win_=t_(win_(1):win_(2));
            peaktimes_subjlvl{cc}{ci}(ss)=t_win_(peaklocs_sorted_(1));
            % for ee=1:128
            %     [peakvals_,peaklocs_]=findpeaks(w_(1,win_(1):win_(2),ee)); 
            %     if ~isempty(peaklocs_)
            %         if length(peaklocs_)>1
            %             % sort by maximum
            %             [~,idx_]=sort(peakvals_);
            %             peaklocs_sorted_=peaklocs_(idx_);
            %         else
            %             peaklocs_sorted_=peaklocs_;
            %         end
            %         t_win_=t_(win_(1):win_(2));
            %         peaktimes_subjlvl{cc}{ci}(ss,ee)=t_win_(peaklocs_sorted_(1));
                % end
            % end
        end
    end
end
clear w_ t_ component_tmas_ peakvals_ peaklocs_ peaklocs_sorted_ t_win_ w_avg_ w_avg_rect_

%% -- Visualize peaks picked per subject --
% WILL GENERATE MANY PLOTS POTENTIALLY BE CAREFUL
show_individual_trfs=false;
if show_individual_trfs
% --- layout constraints ---
N_=numel(experiment_conditions);
M_=max(cellfun(@(x) size(x,1), stats_windows));
ncols_=M_*2; % each component window occupies two subplot columns
nrows_=N_;
topo_limits_=[]; % []= auto or some numeric values
if isempty(topo_limits_)
    topo_limits_='absmax';
end
% -------
titles_=cell(N_,M_);

for cc=1:N_
    t_ms_=avg_models(1).t; % [1 x time, ms]
    for ci=1:size(stats_windows{cc},1)
        win_=stats_windows{cc}(ci,:);
        t_start_=t_ms_(win_(1));
        t_end_=t_ms_(win_(2));
        titles_{cc,ci}=sprintf('%s, %0.0f ms-%0.0f ms', ...
            experiment_conditions{cc},t_start_, ...
            t_end_);
    end
end
clear cc ci win_ t_ms_ t_start_ t_end_
%
for ss=1:n_subjs
    figure('Name', ...
        sprintf('subj %d selected electrodes for subject-level peak latency analysis',subjs(ss)), ...
        'Units', 'inches','Position',[1 1 8 6], ...
        'Color','w')
    t=tiledlayout(nrows_,ncols_,'TileSpacing','tight','Padding','tight');
    for cc=1:N_
        for ci=1:M_
            % note : since moving across tiled layout using nexttile, if
            % condition,window pair is not in original stats_windows, we
            % should skip this tile -- ONLY APPLIES WHEN COMPONENT WINDOW
            % NUMBER VARIES WITH CONDITION
            if ci<=numel(selected_electrodes{cc})       
                % get mask for electrodes containing a prominent peak
                prominent_elecs_mask_=selected_electrodes{cc}{ci}(:);
                trf_=squeeze(ind_models(ss,cc).w)'; % [chn x time]
                times_ms_=ind_models(cc).t; % [1 x time, ms]
                ttl_=titles_{cc,ci};
        
                % --- get column indices in the tiled grid ---
                col_butterfly=(ci-1)*2+1;
                col_topo=(ci-1)*2+2;
                tile_idx_but=(cc-1)*ncols_+col_butterfly;
                tile_idx_topo=(cc-1)*ncols_+col_topo;
                
                % --- Butterfly plot ---
                nexttile(tile_idx_but);
                
                % Note: check plot below has correct dimensions
                plot(times_ms_, trf_(prominent_elecs_mask_,:), ...
                    'Color', [0.6 0.6 0.6 0.4],'LineWidth',0.8)
                hold on; grid on
                plot(times_ms_, mean(trf_(prominent_elecs_mask_,:),1),'k','LineWidth',2);
                % label peak picked
                peak_time_=peaktimes_subjlvl{cc}{ci}(ss);
                plot(peak_time_,mean(trf_(prominent_elecs_mask_,times_ms_==peak_time_)), ...
                    'ro')
            
                % shade topoplot window
                ylims_=ylim;
                win_=stats_windows{cc}(ci,:);
                toi_ms_=[times_ms_(win_(1)),times_ms_(win_(2))];
                patch([toi_ms_(1) toi_ms_(2) toi_ms_(2) toi_ms_(1)], ...
                    [ylims_(1) ylims_(1) ylims_(2) ylims_(2)], ...
                    [1 0.8 0.2], 'FaceAlpha',0.15,'EdgeColor','none', ...
                    'HandleVisibility','off');
        
                xlabel('Time (ms)'); ylabel('Amplitude (a.u.)');
                title(ttl_);
                xlim([-100 400]);
                box off;
        
                % --- Topoplot ---
                nexttile(tile_idx_topo);
                toi_mask_=times_ms_>=toi_ms_(1) & times_ms_<=toi_ms_(2);
                % average over component time window
                % mark electrodes included in peak-latency analysis average TRF
                % using emarker
                topo_data_=mean(trf_(:,toi_mask_),2);
                topoplot(topo_data_, chanlocs, ...
                    'maplimits', topo_limits_, ...
                    'electrodes', 'on', ...
                    'emarker2', {find(prominent_elecs_mask_), 'x', 'k', 1, 0.1})
                colorbar;
                axis off;
            end
        end
    end
    
    
    title(t, sprintf('subj: %d (min prominence: %0.2f)',subjs(ss), ...
        peaktime_stats.opts.prominence_thresh), 'FontSize',14)
end
end
clear N_ M_ titles_ ncols_ nrows_ cmap_ titles_ prominent_elecs_mask_ ttl_ times_ms_ ylims_ win_ toi_ms_ topo_data_ toi_mask_ topo_data_
clear peak_time_
%% -- Repeated-Measures ANOVA + summary statistics plot of peak latency --
% setup data
peak_lat=nan(n_subjs,length(experiment_conditions));
% for ss=1:n_subjs
for cc=1:length(experiment_conditions)
    peak_lat(:,cc)=peaktimes_subjlvl{cc}{:};
end

% package latencies into table (required by fitrm)
T_peak_lat=array2table(peak_lat,'VariableNames', experiment_conditions);

% define within-subject factor specified as 
% r-by-k table that contains the values of the k within-subject factors.
rm_conditions=table(experiment_conditions', 'VariableNames',{'SpeechRate'});
% fit repeated measures model
rm=fitrm(T_peak_lat,'fast-slow ~ 1', 'WithinDesign',rm_conditions);
%run rm-anova
ranova_result=ranova(rm);
% to interpret results, should confirm sphericity assuption of ranova
mauchly_result=mauchly(rm);
% is p<0.005, should use GG-corrected p-value instead
fprintf('Sphericity: %0.3f\n',0.005<=mauchly_result.pValue)
%% --- scatterplot: peak latency per subject, one fig per condition ---
figure('Name','peak latency statistics summary','Color','w', ...
    'Units', 'inches','Position', [1 1 3 2])
fig_jitter_width_=0.5;
fig_jitter_=2*(fig_jitter_width_)*rand(n_subjs,1)-fig_jitter_width_;
plot(1:length(experiment_conditions),peak_lat+fig_jitter_,'k.'); hold on
boxplot(peak_lat)
xticks(1:length(experiment_conditions));
xticklabels(experiment_conditions);
title('TRF P1 Peak Latency vs Speech Rate')
xlim([0 4])
ylim([140 210])
ylabel('P1 latency (ms)')
clear fig_jitter_

% x-axis: subject index
% y-axis: peak latency (ms) - each point is one electrode; overlapping
% electrodes are jittered horizontally to stay readable
% color: electrodes also colored by their index to give spatial context
% 
% latency_fig.jitter_amt=0.35;
% cmap=parula(128);
% 
% for cc=1:numel(experiment_conditions)
%     n_comp=size(stats_windows{cc},1);
% 
%     fig=figure("Name",experiment_conditions{cc}, ...
%         "Color",'w');
% 
%     for ci= 1:n_comp
%         ax=subplot(n_comp,1,ci);
%         hold(ax,'on');
% 
%         data_=peaktimes_subjlvl{cc}{ci}; % nan where no peak, [n_subj x 128]
% 
%         for ee=1:128
%             x_vals= (1:n_subjs)+(rand(1,n_subjs)-0.5)+latency_fig.jitter_amt;
%             y_vals=data_(:,ee); % TIME SHOULD ALREADY BE IN MS???
%             valid_=~isnan(y_vals);
%             if any(valid_)
%                 scatter(ax,x_vals(valid_),y_vals(valid_),10, ...
%                     cmap(ee,:),'filled','MarkerFaceAlpha',0.45);
%             end
%             % overlay per-subject median across electrodes
%             % med_=median(data_,2,'omitnan')*1000;
%             % plot(ax,1:n_subjs,med_,'k-o',...
%             %     'LineWidth',1.5,'MarkerSize',6, ...
%             %     'MarkerFaceColor','k','DisplayName','Electrode median')
% 
%             % visual stuff
%             xlim(ax,[0.5 n_subjs+0.5]);
%             xticks(ax,1:n_subjs);
%             xlabel(ax, 'Subject');
%             ylabel(ax, 'Peak Latency (ms)');
% 
%             % need to convert to time:
%             win_=stats_windows{cc}(ci,:);
%             win_ms=ind_models(1).t(win_);
%             title(ax, sprintf('%s - Component %d (%0.0f-%0.0f ms)', ...
%                 experiment_conditions{cc},ci,win_ms(1),win_ms(2)))
% 
% 
%         end
%     end
% end
