% plot_trfs -> trf_peak_latency
%% -- select TRF peak-prominence threshold --
% subsequent analyses then restricted to average subject-level TRFs across
% ROI defined by peak-prominence

%TODO: add N1 component to fast component_windows
prominence_thresh=0.05; % or idk??

% preallocate cell array to contain prominences
selected_electrodes=cellfun(@(x) repmat({false(1,128)},1,size(x,1)), ...
    component_windows,'UniformOutput',false); %{1 x conditions}, {1 x components}, [1 x electrodes]
peaktimes_grand=cellfun(@(x) repmat(cell(1,128),size(x,1),1), ...
    component_windows,'UniformOutput',false); %{1 x conditions}, {components x electrodes}
% -- plot topo with selected electrodes marked alongside color-coded
% butterfly 
for cc=1:numel(experiment_conditions)
    w_=avg_models(cc).w;
    % rectify weights before peak-picking
    w_rect_=sign(w_).*w_;
    for ci=1:size(component_windows{cc},1)
        win_=component_windows{cc}(ci,:); % indices, not time 
        % (relative to full lag window)
        t_win_=avg_models(cc).t(win_(1):win_(2));
        for ee=1:128
            [~,peaklocs_]=findpeaks(w_rect_(win_(1):win_(2)),'MinPeakProminence',prominence_thresh);
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


%% -- visualize selected electrode grand-TRFs
% --- layout constraints ---
N_=numel(experiment_conditions);
M_=max(cellfun(@(x) size(x,1), component_windows));
ncols_=M_*2; % each component window occupies two subplot columns
nrows_=N_;
% cmap_=colormap('RdBu'); 
topo_limits_=[]; % []= auto or some numeric values
if isempty(topo_limits_)
    topo_limits_='absmax';
end
% -------
titles_=cell(N_,M_);
for cc=1:N_
    for ci=1:size(component_windows{cc},1)
        titles_{cc,ci}=sprintf('%s, %0.0f ms-%0.0f ms', ...
            experiment_conditions{cc},component_windows{cc}(ci,1), ...
            component_windows{cc}(ci,2));
    end
end
clear cc ci
%
figure('Name', 'selected electrodes for subject-level peak latency analysis')
t=tiledlayout(nrows_,ncols_,'TileSpacing','compact','Padding','compact');
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
            nexttile(tile_idx_but);grid on
            
            % Note: check plot below has correct dimensions
            plot(times_ms_, trf_(prominent_elecs_mask_,:), ...
                'Color', [0.6 0.6 0.6 0.4],'LineWidth',0.8)
            hold on;
            plot(times_ms_, mean(trf_(prominent_elecs_mask_,:),1),'k','LineWidth',2);
            
            % TODO: add component window markers?
            % shade topoplot window
            ylims_=ylim;
            win_=component_windows{cc}(ci,:);
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
                'emarker2', {find(prominent_elecs_mask_), 'x', 'k', 8, 1})
            colorbar;
            axis off;
        end
    end
end

clear N_ M_ titles_ ncols_ nrows_ cmap_ titles_ prominent_elecs_mask_ ttl_ times_ms_ ylims_ win_ toi_ms_ topo_data_ toi_mask_ topo_data_
title(t, sprintf('min prominence: %0.2f',prominence_thresh), 'FontSize',14)
%% --- identify peaks in ind_model weights ---
% for each subject/electrode within predefined component windows

% TODO: restrict to subset of electrodes shown above
%TODO: permutation test on final peaktimes


peaktimes_subjlvl=cellfun(@(x) repmat({nan(n_subjs,128)},1,size(x,1)), ...
    component_windows,'UniformOutput',false); % {1 x conditions}, {1 x components}, [subjs x electrodes]

for ss=1:n_subjs
    for cc=1:numel(experiment_conditions)
        w_=ind_models(ss,cc).w;
        t_=ind_models(ss,cc).t;


        for ci = 1:size(component_windows{cc},1)
            win_=component_windows{cc}(ci,:); % indices, not time
      
            for ee=1:128
                [peakvals_,peaklocs_]=findpeaks(w_(1,win_(1):win_(2),ee)); 
                if ~isempty(peaklocs_)
                    if length(peaklocs_)>1
                        % sort by maximum
                        [~,idx_]=sort(peakvals_);
                        peaklocs_sorted_=peaklocs_(idx_);
                    else
                        peaklocs_sorted_=peaklocs_;
                    end
                    t_win_=t_(win_(1):win_(2));
                    peaktimes_subjlvl{cc}{ci}(ss,ee)=t_win_(peaklocs_sorted_(1));
                end
            end
        end
    end
end
clear w_ t_ component_tmas_ peakvals_ peaklocs_ peaklocs_sorted_ t_win_
%% --- scatterplot: peak latency per subject, one fig per condition ---
% x-axis: subject index
% y-axis: peak latency (ms) - each point is one electrode; overlapping
% electrodes are jittered horizontally to stay readable
% color: electrodes also colored by their index to give spatial context

latency_fig.jitter_amt=0.35;
cmap=parula(128);

for cc=1:numel(experiment_conditions)
    n_comp=size(component_windows{cc},1);

    fig=figure("Name",experiment_conditions{cc}, ...
        "Color",'w');

    for ci= 1:n_comp
        ax=subplot(n_comp,1,ci);
        hold(ax,'on');
        
        data_=peaktimes_subjlvl{cc}{ci}; % nan where no peak, [n_subj x 128]

        for ee=1:128
            x_vals= (1:n_subjs)+(rand(1,n_subjs)-0.5)+latency_fig.jitter_amt;
            y_vals=data_(:,ee); % TIME SHOULD ALREADY BE IN MS???
            valid=~isnan(y_vals);
            if any(valid)
                scatter(ax,x_vals(valid),y_vals(valid),10, ...
                    cmap(ee,:),'filled','MarkerFaceAlpha',0.45);
            end
            % overlay per-subject median across electrodes
            % med_=median(data_,2,'omitnan')*1000;
            % plot(ax,1:n_subjs,med_,'k-o',...
            %     'LineWidth',1.5,'MarkerSize',6, ...
            %     'MarkerFaceColor','k','DisplayName','Electrode median')

            % visual stuff
            xlim(ax,[0.5 n_subjs+0.5]);
            xticks(ax,1:n_subjs);
            xlabel(ax, 'Subject');
            ylabel(ax, 'Peak Latency (ms)');
            
            % need to convert to time:
            win_=component_windows{cc}(ci,:);
            win_ms=ind_models(1).t(win_);
            title(ax, sprintf('%s - Component %d (%0.0f-%0.0f ms)', ...
                experiment_conditions{cc},ci,win_ms(1),win_ms(2)))


        end
    end
end
%% --- do some statistical test across subjects ---
% should compare across conditions, within subjects



