clearvars -except user_profile boxdir_mine boxdir_lab
%note: plotting sep conditions gives errors when constructing avg models
%AND ind models havve 3 cols even though should be 1...
%% plotting params
% TODO: take automatic tile bs out of main weight-plotting helper function
close all
subjs=[2:7,9:22];
plot_chns='all';
separate_conditions=true; %NOTE: not operator required when 
    % initializing config_trf since technically it expects 
    % "do_lambda_optimization" as argument 
    % ignoring the false case rn sincee buggy but not a priority but should
    % fix
n_subjs=numel(subjs);
plot_individual_weights=false;
plot_avg_weights=true;
if separate_conditions
    conditions={'fast','og','slow'};
else
    conditions={'all conditions'};
end
n_cond=numel(conditions);
%% Main script
for ss=1:n_subjs
    subj=subjs(ss);
    fprintf('loading subj %d data...\n',subj)
    
    preprocess_config=config_preprocess(subj);
    trf_config=config_trf(subj,~separate_conditions,preprocess_config);
    % note we're expecting model coming out to be (1,3) regardless of
    % condtion specified but ideally load individual model should return
    % (1,1) struct
    ind_models(ss,:)=load_individual_model(trf_config);
    if plot_individual_weights
        for cc=1:n_cond
            title_str=sprintf('subj: %d - chns %s - %s',trf_config.subj, ...
                    num2str(plot_chns),conditions{cc});
            % plot_model_weights(ind_models(ss,:),trf_config,plot_chns)
            figure
            mTRFplot(ind_models(ss,cc),'trf','all',plot_chns);
            title(title_str)
        end
    end

end

%% Plot average weights
if plot_avg_weights && n_subjs>1
    avg_models=construct_avg_models(ind_models);
    for cc=1:n_cond
         title_str=sprintf('subj-avg TRF - chns: %s - condition: %s', ...
                num2str(plot_chns),conditions{cc});
        figure
        mTRFplot(avg_models(1,cc),'trf','all',plot_chns);
        title(title_str)
    end
end
%% estimate snr overall
if numel(subjs)>1
    snr=estimate_snr(avg_models);
    for cc=1:3
        %TODO: take conditions cell out
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
    for cc=1:n_cond
         plot(1:n_subjs,snr_per_subj(:,cc))
         hold on
    end
    legend(conditions)
    xlabel('n subjects')
ylabel('snr')
end
%% plot topos at particular latency

plot_topos=false;
% note: we should perhaps generate a topo-movie across entire timeframe..

if plot_topos
    global boxdir_mine
    loc_file=sprintf("%s/analysis/128chanlocs.mat",boxdir_mine);
    load(loc_file);
    topo_latencies=[54 164]; % in ms
    if plot_avg_weights && n_subjs>1
        for tt=1:numel(topo_latencies)
            for cc_topo=1:n_cond
                % finding time closest to those latencies to plot
                [~,t_ii]=min(abs(avg_models(cc_topo).t-topo_latencies(tt)));
                figure
                topoplot(avg_models(1,cc_topo).w(1,t_ii,:),chanlocs)
                title(sprintf(['subject-averaged trf model weights %0.1f ms, ' ...
                    'condition: %s'] ...
                    ,avg_models(1,cc_topo).t(t_ii),conditions{cc_topo}));
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
prom_thresh=0;
t_range=avg_models(1).t(time_range_idx);
n_electrodes=size(avg_models(1).w,3);
% time range considered "reliable" evoked response
evoked_tlims=[0, 400];
evoked_range_idx=find(avg_models(1).t>evoked_tlims(1)& ...
    avg_models(1).t<evoked_tlims(2));
pk_locs=cell(n_cond,n_electrodes);
for cc=1:n_cond
    w_=squeeze(avg_models(cc).w(1,time_range_idx,:));
    w_=w_';
    % filter peaks by std of weights (in 0->400 ms time range)  
    w_std_=squeeze(std(avg_models(cc).w(1,evoked_range_idx,:),[],2));
    for ee=1:n_electrodes
        [~, locs_]=findpeaks(w_(ee,:), ...
            'MinPeakProminence',prom_thresh, ...
            'MinPeakHeight',2*w_std_(ee));
        % proms_=proms_/std(proms_);
        % pk_locs{cc,ee}=locs_(proms_>prom_thresh);
        if any(locs_)
            pk_locs{cc,ee}=locs_;
        end
    end
    clear w_ locs_ proms_ w_std_
end
conditions={'fast','og','slow'};

for cc=1:n_cond
    title_str=sprintf('subj-avg TRF - chns: %s - condition: %s', ...
            num2str(plot_chns),conditions{cc});
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
single_pk_electrodes_idx=find(single_pk_electrodes);
n_single_peak_electrodes=sum(single_pk_electrodes);
fprintf('number of electrodes with single peak across conditions:%d\n', ...
    n_single_peak_electrodes)
% check that number of peaks matches across conditions for each electrode
max_pkcount=max(pk_counts(:));
fprintf('max peakcount: %d\n',max_pkcount)
%% topo of electrodes with distinct peak
global boxdir_mine
loc_file=sprintf("%s/analysis/128chanlocs.mat",boxdir_mine);
load(loc_file);
figure
topoplot([],chanlocs,'electrodes','on','style','blank', ...
    'plotchans',single_pk_electrodes_idx,'emarker',{'o','r',5,1});
title('electrodes with distinct peaks')
%% visualize difference in latency across conditions
pk_latencies=nan(n_cond,n_single_peak_electrodes);
for cc=1:n_cond
    pk_latencies(cc,:)=t_range([pk_locs{cc,single_pk_electrodes_idx}]);
end
% add jitter to minimize overlapping lines
rng(1);
yjitter=(1000/fs)*repmat(rand([1,n_single_peak_electrodes]),n_cond,1);
% sort them for pretty colors
[~,sortI_]=sort(pk_latencies(1,:));
figure
plot(1:n_cond,pk_latencies(:,sortI_)+yjitter(:,sortI_))
colormap(jet(n_single_peak_electrodes))
colororder(jet(n_single_peak_electrodes))
xticks(1:n_cond);
xticklabels(conditions);
xlabel('condition');
ylabel('latency (ms)')
title('TRF peak latency (+jitter) across conditions')
hold off
clear sortI_
% histograms of difference relative to og
diff_pk_latency=nan(n_cond-1,n_single_peak_electrodes);
%dum counter
cc_=1;
diff_labels=cell(n_cond-1);
for cc=1:2:n_cond
    diff_pk_latency(cc_,:)=pk_latencies(2,:)-pk_latencies(cc,:);
    diff_labels{cc_}=sprintf('%s-%s',conditions{2},conditions{cc});
    cc_=cc_+1;
end
clear cc_
figure
boxplot(diff_pk_latency')
xticklabels(diff_labels)

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

% function plot_model_weights(model,trf_config,chns)
% %TODO: use a switch-case to handle plotting a particular channel, the
%         %best channel, or all channels... needs to chance title string
%         %format indicator
%     arguments
%         model struct
%         trf_config (1,1) struct
%         chns = 'all'
%     end
%     n_subjs=size(model(1).w,1);
%     if trf_config.separate_conditions
%         n_conditions=size(model,2);
%         fprintf('plotting condition-specific TRFs...\n ')
%     else
%         n_conditions=1;
%         fprintf("NOTE: line above this will bypass bad config handling in " + ...
%             "separate_conditions=false case but that also means the issue " + ...
%             "will be hidden and still occur so needs to be fixed directly...\n")
%     end
% 
%     for cc=1:n_conditions
%         if ~isfield(model,'avg')
%             fprintf('plotting individual model weights...\n')
%             %plot individual subject weights            
%             figure
%             mTRFplot(model(1,cc),'trf','all',chns);
% 
%         else
%             % assume we want to average out the weights and plot them
%             % compile weights into single model struct for current
%             % condition
%             fprintf('plotting subject-averaged model weights...\n')
%             avg_models=construct_avg_models(model);
%             figure
%             mTRFplot(avg_models(1,cc),'trf','all',chns);
%             ylim([-.4,.55])
% 
% 
%         end
%     end
% end

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
    disp(['TODO: fix bug - this function is return trf_config ...' ...
        ' also will be a problem in analysis script'])
    if exist(trf_config.trf_model_path,'file')
        % model_checkpoint=load_checkpoint(trf_config.trf_model_path,trf_config);
        % note: load_checkpoint is causing more pain than it's worth... if
        % we fix it at some point later we can continue using it but right
        % now just assuming only relevant differences in configs is wether
        % lambda optimization was done (step 1) or not (step 2 - separate conditions)
        data_idx=trf_config.separate_conditions+1;
        temp=load(trf_config.trf_model_path);
        if ismember('model',fieldnames(temp))
            fprintf('model found in %s\n',trf_config.trf_model_path)
            model=temp.model(data_idx,:);
        else
            fprintf('no model found in %s\n',trf_config.trf_model_path)
        end
    end
end

