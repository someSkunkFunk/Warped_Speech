% trf_peak_latency
% identify peak in ind_model weights for each subject electrode at
% component_windows


peaktimes_subjlvl=cellfun(@(x) repmat({nan(n_subjs,128)},1,size(x,1)), ...
    component_windows,'UniformOutput',false);

for ss=1:n_subjs
    for cc=1:numel(experiment_conditions)
        w_=ind_models(ss,cc).w;
        t_=ind_models(ss,cc).t;
        % NOTE: should add a warning that this won't work with microstate
        % analysis... not sure we'll come back to it though
        % component window indices are relative to tbounds defined in basic
        % component analysis, correct for it here
        % component_tmas_=t_>basic_component_analysis.tbounds(1)&t_<basic_component_analysis.tbounds(2);
        % w_=w_(:,component_tmas_,:);
        % t_=t_(component_tmas_);

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



