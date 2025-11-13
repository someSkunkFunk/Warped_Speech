clearvars -except user_profile boxdir_mine boxdir_lab
%note: plotting sep conditions gives errors when constructing avg models
%AND ind models havve 3 cols even though should be 1...
%% plotting params
% TODO: take automatic tile bs out of main weight-plotting helper function
close all
% fast-slow subjs:
subjs=[2:7,9:22];
% subjs=[2:3]
% best fast-slow subjs: 
% subjs=[9,12]; 
% reg-irreg subjects:
% subjs=[23,96,97,98];
% subjs=[96:98];
% for reg-irreg:
mtrf_plot_chns=[1 97 98 65 66 87 86 85 84 83 107 62];
% mtrf_plot_chns='all';
n_subjs=numel(subjs);
script_config.show_individual_weights=false;
script_config.show_avg_weights=true;
script_config.show_topos=false;
script_config.show_snr=false;
script_config.show_tuning_curves=false;
script_config.analyze_pk_latencies=false;

%%% TODO: improve PSDs to include
%%% TODO: add GFP plots
trf_fig_param.t_lims=[-100 500];
%fastslow
trf_fig_param.ylims=[-.5 .6];
%regirreg
% trf_fig_param.ylims=[-1 1];
trf_fig_param.lw=2; %linewidth in plot
trf_fig_param.leg_lw=6; % linewidth in legend
trf_fig_param.fz=28; % fontsize
trf_fig_param.title_fz=28;
% fast is red, black is slow, green is og
trf_fig_param.condition_colors=struct('slow',[255 0 0]./255,'original',[1 150 55]./206, ...
    'fast',[0 0 0],'reg',[255 0 0]./255, ...
    'irreg',[0 0 0]);
% for fast slow
% trf_fig_param.r_thresh=[0.03]; % leave empty if all chns should be kept for sbj-averaged plot

% for reg-irreg
trf_fig_param.r_thresh=[];
trf_fig_param.stack=true;
% x y width height - set to inches in figure
trf_fig_param.pos=[0 0 8 8];
%% Main script

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
    if ~isempty(trf_fig_param.r_thresh)
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
        if ~isempty(trf_fig_param.r_thresh)
            sz_=[n_subjs, size(rs_)];
            rs=nan(sz_);
        end
        clear model_fields_ sz_
    end
    ind_models(ss,:)=model_; clear model_
    if ~isempty(trf_fig_param.r_thresh)
        rs(ss,:,:)=rs_; clear rs_
    end
    % function ends here
    
    configs(ss).preprocess_config=preprocess_config;
    configs(ss).trf_config=trf_config;
    clear preprocess_config trf_config
end
experiment_conditions=configs(end).trf_config.conditions;
if script_config.show_topos
    load(loc_file);
end
% determine which channels to keep based on r_thresh
if ~isempty(trf_fig_param.r_thresh)
    fprintf('filtering chns based on r_thresh=%0.3f...\n',trf_fig_param.r_thresh)   
    % average out subjects
    chns_m=squeeze(mean(rs,1))>trf_fig_param.r_thresh;
    disp('chns remaining:')
    disp(sum(chns_m,2));
    if any(sum(chns_m,2)==0)
        warning('empty chns array will cause mtrfplot to show all the channels')
    end
end
%% plot weights for individual subject
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
                % else
                %     fprintf(['filtering channels plotted based ' ...
                %         'on r_thresh=%d...\n'],r_thresh)
                % end
                title(title_str_)
                title(title_str_,'FontSize',trf_fig_param.fz)
                set(h_,'LineWidth',trf_fig_param.lw)
            else
                fprintf('nothing to plot for %s\n', configs(ss).trf_config.conditions{cc})
            end
            clear title_str_ h_ model_
        end
    end
end
%% Plot subj-averaged weights
if script_config.show_avg_weights
    avg_models=construct_avg_models(ind_models);
    for cc=1:numel(configs(end).trf_config.conditions)
        model_=avg_models(1,cc);
        if ~isempty(model_.w)

            if cc>1&&trf_fig_param.stack
                % skip adding fig
            else
                figure('Units','inches','Position',trf_fig_param.pos)
            end

            if isempty(trf_fig_param.r_thresh)
                h_=mTRFplot(model_,'trf','all',mtrf_plot_chns);
            else
                h_=mTRFplot(model_,'trf','all',find(chns_m(cc,:)));
            end
            set(h_,'LineWidth',trf_fig_param.lw, ...
                'Color',trf_fig_param.condition_colors.(experiment_conditions{cc}))
            if cc>1&&trf_fig_param.stack
                % skip adding title
            elseif trf_fig_param.stack
                % add title once
                title_str=sprintf('subj-avg TRF - chns: %s ',num2str(mtrf_plot_chns));
                title(title_str,'FontSize',trf_fig_param.title_fz)
            else
                title_str=sprintf('subj-avg TRF - chns: %s - condition: %s', ...
                    num2str(mtrf_plot_chns),experiment_conditions{cc});
                title(title_str,'FontSize',trf_fig_param.fz)
            end
            xlim(trf_fig_param.t_lims)
            ylim(trf_fig_param.ylims)
            ylabel('Amplitude (a.u.)')
            grid off
            set(gca,'FontSize',trf_fig_param.fz)
            if trf_fig_param.stack,hold on,end
        else
            % useful for pilot subjects that only listen to a single
            % condition...
            fprintf('nothing to plot for %s\n',configs(ss).trf_config.conditions{cc})
        end
    end
    if trf_fig_param.stack
        % otherwise title has condition name
        hold off
        legend_helper(gca,experiment_conditions,trf_fig_param.condition_colors);
        % set(lh,'LineWidth',trf_fig_param.leg_lw) % ended up not being
        % useful since it just makes box around legend thicker
        
    end

    clear  h_ lg_
end
% %% estimate snr overall
% if numel(subjs)>1&&script_config.show_snr
%     snr=estimate_snr(avg_models);
%     for cc=1:3
%         fprintf('condition %d rms snr estimate: %0.3f\n',cc,snr(cc))
%     end
% end
%% estimate snr per subject (simplified)
if n_subjs>1&&script_config.show_snr
    snr_per_subj=nan(n_subjs,3);
    for ss=1:n_subjs
        subset_avg_model=construct_avg_models(ind_models(1:ss,:));
        %TODO: need to transpose?
        snr_per_subj(ss,:)=estimate_snr(subset_avg_model);
    end
    figure
    for cc=1:numel(configs(end).trf_config.conditions)
         plot(1:n_subjs,snr_per_subj(:,cc))
         hold on
    end
    legend(configs(end).trf_config.conditions)
    xlabel('n subjects')
ylabel('snr')
end
%% plot topos at particular latency

% note: we should perhaps generate a topo-movie across entire timeframe..

if script_config.show_topos
    topo_latencies=[54 164]; % in ms
    if script_config.show_avg_weights && n_subjs>1
        for tt=1:numel(topo_latencies)
            for cc_topo=1:numel(configs(end).trf_config.conditions)
                % finding time closest to those latencies to plot
                [~,t_ii]=min(abs(avg_models(cc_topo).t-topo_latencies(tt)));
                figure
                topoplot(avg_models(1,cc_topo).w(1,t_ii,:),chanlocs)
                title(sprintf(['subject-averaged trf model weights %0.1f ms, ' ...
                    'condition: %s'] ...
                    ,avg_models(1,cc_topo).t(t_ii),configs(end).trf_config.conditions{cc_topo}));
                colorbar
            end
        end
    end
end
%% TRF component latency analysis
%% plot average TRF peak latencies across electrodes - use findpeaks 
if script_config.analyze_pk_latencies
    % within range of post onset latencies
    t_seek=[100 250]; %ms range within which to find peaks
    time_range_idx=find(avg_models(1).t>t_seek(1)&avg_models(1).t<t_seek(2));
    % filter peaks by prominence
    % prom_thresh=0;
    t_range=avg_models(1).t(time_range_idx);
    n_electrodes=size(avg_models(1).w,3);
    % time range considered "reliable" evoked response
    evoked_tlims=[0, 400];
    evoked_range_idx=find(avg_models(1).t>evoked_tlims(1)& ...
        avg_models(1).t<evoked_tlims(2));
    pk_locs=cell(numel(configs(end).trf_config.conditions),n_electrodes);
    pk_locs_config=struct('MinPeakProminence',0,'MinPeakHeight_std_thresh',2);
    disp('looking for peaky electrodes with params:')
    disp(pk_locs_config)
    for cc=1:numel(configs(end).trf_config.conditions)
        w_=squeeze(avg_models(cc).w(1,time_range_idx,:));
        w_=w_';
        % filter peaks by std of weights (in 0->400 ms time range)  
        w_std_=squeeze(std(avg_models(cc).w(1,evoked_range_idx,:),[],2));
        
        for ee=1:n_electrodes
            fprintf('electrode #%d...\n',ee)
            [~, locs_]=findpeaks(w_(ee,:), ...
                'MinPeakProminence',pk_locs_config.MinPeakProminence, ...
                'MinPeakHeight',pk_locs_config.MinPeakHeight_std_thresh*w_std_(ee));
            % proms_=proms_/std(proms_);
            % pk_locs{cc,ee}=locs_(proms_>prom_thresh);
            if any(locs_)
                pk_locs{cc,ee}=locs_;
            end
        end
        clear w_ locs_ proms_ w_std_
    end
    % conditions={'fast','og','slow'};
    
    for cc=1:numel(configs(end).trf_config.conditions)
        title_str=sprintf('subj-avg TRF - chns: %s - condition: %s', ...
                num2str(mtrf_plot_chns),experiment_conditions{cc});
        figure
        h_=mTRFplot(avg_models(1,cc),'trf','all',mtrf_plot_chns);
        set(h_,'LineWidth',trf_fig_param.lw)
        hold on
        for ee=1:n_electrodes
            locs_=pk_locs{cc,ee};
            % trim pre-onset stuff first, then index peaks
            pks_t_=avg_models(1,cc).t(time_range_idx);
            pks_t_=pks_t_(locs_);
            pks_w_=squeeze(avg_models(1,cc).w(1,time_range_idx,ee));
            pks_w_=pks_w_(locs_);
            stem(pks_t_,pks_w_);
            clear locs_
        end
        xlim(evoked_tlims)
        title(title_str,'FontSize',trf_fig_param.fz)
        clear h_
    end
    % check that there is one peak per electrode/condition
    pk_counts=cellfun(@numel, pk_locs);
    cond_pkcounts_match=all(pk_counts==pk_counts(1,:),1);
    fprintf('all electrodes have same num of peaks? ->%d\n',all(cond_pkcounts_match))
    single_pk_electrodes=cond_pkcounts_match&pk_counts(1,:)==1;
    n_single_peak_electrodes=sum(single_pk_electrodes);
    fprintf('number of electrodes with single peak across conditions:%d\n', ...
        n_single_peak_electrodes)
    if any(single_pk_electrodes)
        single_pk_electrodes_idx=find(single_pk_electrodes);
    
    
        % check that number of peaks matches across conditions for each electrode
        max_pkcount=max(pk_counts(:));
        fprintf('max peakcount: %d\n',max_pkcount)
        %% topo of electrodes with distinct peak
        
        figure
        topoplot([],chanlocs,'electrodes','on','style','blank', ...
            'plotchans',single_pk_electrodes_idx,'emarker',{'o','r',5,1});
        title('electrodes with distinct peaks')
        %% visualize difference in latency acrossl conditions
        pk_latencies=nan(numel(configs(end).trf_config.conditions),n_single_peak_electrodes);
        for cc=1:numel(configs(end).trf_config.conditions)
            pk_latencies(cc,:)=t_range([pk_locs{cc,single_pk_electrodes_idx}]);
        end
        % add jitter to minimize overlapping lines
        rng(1);
        yjitter=(1000/avg_models(1).fs)*repmat(rand([1,n_single_peak_electrodes]),numel(configs(end).trf_config.conditions),1);
        % sort them for pretty colors
        [~,sortI_]=sort(pk_latencies(1,:));
        figure
        plot(1:numel(configs(end).trf_config.conditions),pk_latencies(:,sortI_)+yjitter(:,sortI_))
        colormap(jet(n_single_peak_electrodes))
        colororder(jet(n_single_peak_electrodes))
        xticks(1:numel(configs(end).trf_config.conditions));
        xticklabels(configs(end).trf_config.conditions);
        xlabel('condition');
        ylabel('latency (ms)')
        title('TRF peak latency (+jitter) across conditions')
        hold off
        clear sortI_
        % histograms of difference relative to og
        diff_pk_latency=nan(numel(configs(end).trf_config.conditions)-1,n_single_peak_electrodes);
        %dum counter
        cc_=1;
        diff_labels=cell(numel(configs(end).trf_config.conditions)-1);
        for cc=1:2:numel(configs(end).trf_config.conditions)
            diff_pk_latency(cc_,:)=pk_latencies(2,:)-pk_latencies(cc,:);
            diff_labels{cc_}=sprintf('%s-%s',experiment_conditions{2},experiment_conditions{cc});
            cc_=cc_+1;
        end
        clear cc_
        figure
        boxplot(diff_pk_latency')
        xticklabels(diff_labels)
        ylabel('\Delta(latency) (ms)')
        title('difference in latency across reliable electrodes')
    else
        disp('*****************NO PEAKS FOUND :( *****************')
    end
end
%% Helpers
function lh=legend_helper(ax,color_labels,color_rgbs)
    % condition_colors is struct with condition names 
    line_objs=cell2struct(cell(size(color_labels))',color_labels);
    legend_lines=[];
    for ff=1:numel(color_labels)
        line_objs.(color_labels{ff})=findobj(ax,'Type','Line','Color',color_rgbs.(color_labels{ff}));
        legend_lines(ff)=line_objs.(color_labels{ff})(1);
    end
    lh=legend(legend_lines,color_labels);
end

function snr_plot(snr_per_subj)
    [n_subjs,n_cond]=size(snr_per_subj);
    for cc=1:n_cond
        plot(1:n_subjs,snr_per_subj(ss,cc));
        
    end
end

function snr=estimate_snr(avg_models,noise_window,signal_window)
% assumes (1,3) avg models given
arguments
    avg_models (1,3) struct
    noise_window (1,2) = [-200, 0]
    signal_window (1,2) = [100,300]
end
snr=nan(1,3);
n_conditions=size(avg_models,2);

for cc=1:n_conditions
    noise_mask=avg_models(1,cc).t<max(noise_window)&avg_models(1,cc).t>min(noise_window);
    signal_mask=avg_models(1,cc).t<max(signal_window)&avg_models(1,cc).t>min(signal_window);
    snr(cc)=rms(avg_models(1,cc).w(signal_mask))/rms(avg_models(1,cc).w(noise_mask));
end

end

function avg_model=construct_avg_models(ind_models)
    [n_subjs,n_conditions]=size(ind_models);
    [~,n_weights,n_chans]=size(ind_models(1).w);
    
    avg_model=struct();
    model_fields=fieldnames(ind_models(1,1));
    warning(['NOTE: avg model below will only have correct average weights',...
    'other fields which may vary at individual subject level could',...
    'be incorrect...\n'])
    for cc=1:n_conditions
        W_stack=nan(n_subjs,n_weights,n_chans);
        for ss=1:n_subjs
            if ~isempty(ind_models(ss,cc).w)
                W_stack(ss,:,:)=ind_models(ss,cc).w;
            else
                warning('no weights for ss=%d, cc=%d',ss,cc)
            end
        end
        for ff=1:numel(model_fields)
            ff_field=model_fields{ff};
            if strcmp(ff_field,'w')
                if all(isnan(W_stack))
                    avg_model(1,cc).(ff_field)=[];
                else
                    % can still compute mean weights if not all the
                    % subjects are missing conditioins
                    avg_model(1,cc).(ff_field)=mean(W_stack,1,"omitnan") ;
                end
            elseif strcmp(ff_field,'b')
                % safe to ignore bias... I think?
                continue
            else
                avg_model(1,cc).(ff_field)=ind_models(1,cc).(ff_field);
            end
        end
    %add dummy field to distinguish from native trf toolbox model
    avg_model(1,cc).avg=true;
    end
    
end

function model=load_individual_model(trf_config)

end

