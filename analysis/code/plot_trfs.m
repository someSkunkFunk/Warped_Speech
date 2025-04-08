clearvars -except user_profile boxdir_mine boxdir_lab
%% plotting params
%

subjs=2:17;
plot_chns=85;
separate_conditions=true; %NOTE: not operator required when 
    % initializing config_trf since technically it expects 
    % "do_lambda_optimization" as argument 
    % ignoring the false case rn sincee buggy but not a priority but should
    % fix
n_subjs=numel(subjs);
plot_individual_weights=false;
plot_avg_weights=true;
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
        plot_model_weights(ind_models(ss,:),trf_config,plot_chns)
    end

end
%% Plot average weights
if plot_avg_weights
    avg_models=construct_avg_models(ind_models);
    plot_model_weights(avg_models,trf_config,plot_chns)
end
%% estimate snr
snr=estimate_snr(avg_models);
for cc=1:3
    %TODO: take conditions cell out
    fprintf('condition %d rms snr estimate: %0.3f\n',cc,snr(cc))
end
%% Helpers
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

function plot_model_weights(model,trf_config,chns)
%TODO: use a switch-case to handle plotting a particular channel, the
        %best channel, or all channels... needs to chance title string
        %format indicator
    arguments
        model struct
        trf_config (1,1) struct
        chns = 'all'
    end
    n_subjs=size(model,1);
    if trf_config.separate_conditions
        n_conditions=size(model,2);
        fprintf('plotting condition-specific TRFs...\n ')
    else
        n_conditions=1;
        fprintf("NOTE: line above this will bypass bad config handling in " + ...
            "separate_conditions=false case but that also means the issue " + ...
            "will be hidden and still occur so needs to be fixed directly...\n")
    end
    conditions={'fast','og','slow'};
    for cc=1:n_conditions
        if ~isfield(model,'avg')
            fprintf('plotting individual model weights...\n')
            %plot individual subject weights            
            title_str=sprintf('subj: %d - chns %s - %s',trf_config.subj, ...
                num2str(chns),conditions{cc});
            figure
            mTRFplot(model(1,cc),'trf','all',chns);
            title(title_str)
        else
            % assume we want to average out the weights and plot them
            % compile weights into single model struct for current
            % condition
            fprintf('plotting subject-averaged model weights...\n')
            avg_models=construct_avg_models(model);

            % NOTE: repetitive code below could be consolidated across
            % single condition vs separate condition trf cases...
            title_str=sprintf('avg TRF - chns: %s - condition: %s - n subjs: %d', ...
                num2str(chns),conditions{cc},n_subjs);
            figure
            mTRFplot(avg_models(1,cc),'trf','all',chns);
            title(title_str)
            
        
        end
    end
end

function avg_model=construct_avg_models(ind_models)
    [n_subjs,n_weights,n_chans]=size(ind_models(1,1).w);
    n_conditions=size(ind_models,2);
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
                avg_model(1,cc).(ff_field)=mean(W_stack,1);
            else
                avg_model(1,cc).(ff_field)=ind_models(1,cc).(ff_field);
            end
        end
    %TODO: better way to structure this?
    avg_model(1,cc).avg=true;
    end
    
end

function model=load_individual_model(trf_config)
    disp(['TODO: fix bug - this function is return trf_config ...' ...
        ' also will be a problem in analysis script'])
    if exist(trf_config.trf_model_path,'file')
        model_checkpoint=load_checkpoint(trf_config.trf_model_path,trf_config);
        if ismember('model',fieldnames(model_checkpoint))
            fprintf('model found in %s\n',trf_config.trf_model_path)
            model=model_checkpoint.model;
        else
            fprintf('no model found in %s\n',trf_config.trf_model_path)
        end
    end
end

