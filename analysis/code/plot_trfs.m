clearvars -except user_profile boxdir_mine boxdir_lab
%note: plotting sep conditions gives errors when constructing avg models
%AND ind models havve 3 cols even though should be 1...
%% plotting params
% TODO: take automatic tile bs out of main weight-plotting helper function
close all
% subjs=[2:7,9:22];
% subjs=[9,12];
subjs=[96,98]
plot_chns='all';
n_subjs=numel(subjs);
plot_config.show_individual_weights=true;
plot_config.show_avg_weights=true;
plot_config.show_topos=true;
plot_config.show_snr=false;
plot_config.show_tuning_curves=false;

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
    if ~plot_config.show_tuning_curves
        close(gcf)
    end
    clear do_nulltest
    % TODO: make function
    S_=load_checkpoint(trf_config);
    model_=S_.model;
    clear S_
    if ss==1
        %preallocate
        model_fields_=fieldnames(model_);
        sz_=[size(model_),size(model_fields_)];
        sz_(1)=n_subjs;
        ind_models=cell2struct(cell(sz_),model_fields_,3);
        clear model_fields_ sz_
    end
    ind_models(ss,:)=model_;
    clear model_
    % function ends here
    
    configs(ss).preprocess_config=preprocess_config;
    configs(ss).trf_config=trf_config;
    clear preprocess_config trf_config
end

if plot_config.show_topos
    load(loc_file);
end
%% plot weights for individual subject
if plot_config.show_individual_weights
    for ss=1:n_subjs
        for cc=1:numel(configs(ss).trf_config.conditions)
            title_str_=sprintf('subj: %d - chns %s - %s',configs(ss).trf_config.subj, ...
                    num2str(plot_chns),configs(ss).trf_config.conditions{cc});
            % plot_model_weights(ind_models(ss,:),trf_config,plot_chns)
            figure
            mTRFplot(ind_models(ss,cc),'trf','all',plot_chns);
            title(title_str_)
            clear title_str_
        end
    end
end
%% Plot subj-averaged weights
if plot_config.show_avg_weights
    t_lims_=[-400 600];
    ylims_=[-.5 .7];
    avg_models=construct_avg_models(ind_models);
    for cc=1:numel(configs(end).trf_config.conditions)
         title_str=sprintf('subj-avg TRF - chns: %s - condition: %s', ...
                num2str(plot_chns),configs(end).trf_config.conditions{cc});
        figure
        h=mTRFplot(avg_models(1,cc),'trf','all',plot_chns);
        title(title_str,'FontSize',16)
        set(h,'LineWidth',0.25)
        xlim(t_lims_)
        ylim(ylims_)
    end
    clear t_lims_
end
%% estimate snr overall
if numel(subjs)>1&&plot_config.show_snr
    snr=estimate_snr(avg_models);
    for cc=1:3
        fprintf('condition %d rms snr estimate: %0.3f\n',cc,snr(cc))
    end
end
%% estimate snr per subject (simplified)
if n_subjs>1
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

if plot_config.show_topos
    topo_latencies=[54 164]; % in ms
    if plot_config.show_avg_weights && n_subjs>1
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
            num2str(plot_chns),configs(end).trf_config.conditions{cc});
    figure
    mTRFplot(avg_models(1,cc),'trf','all',plot_chns);
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
    title(title_str)
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
        diff_labels{cc_}=sprintf('%s-%s',configs(end).trf_config.conditions{2},configs(end).trf_config.conditions{cc});
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

%% Helpers

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
    W_stack=nan(n_subjs,n_weights,n_chans);
    avg_model=struct();
    model_fields=fieldnames(ind_models(1,1));
    fprintf(['NOTE: avg model below will only have correct average weights',...
    'other fields which may vary at individual subject level could',...
    'be incorrect...\n'])
    for cc=1:n_conditions
        for ss=1:n_subjs
            W_stack(ss,:,:)=ind_models(ss,cc).w;
        end
        for ff=1:numel(model_fields)
            ff_field=model_fields{ff};
            if strcmp(ff_field,'w')
                avg_model(1,cc).(ff_field)=mean(W_stack,1) ;
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

