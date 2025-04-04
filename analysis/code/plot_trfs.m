clearvars -except user_profile boxdir_mine boxdir_lab
%%
%plotting params

%%
subjs=16;
plot_chns=85;
for ss=1:numel(subjs)
    subj=subjs(ss);
    fprintf('loading subj %d data...\n',subj)
    separate_conditions=false; %NOTE: not operator required when 
    % initializing config_trf since technically it expects 
    % "do_lambda_optimization" as argument 
    preprocess_config=config_preprocess(subj);
    trf_config=config_trf(subj,~separate_conditions,preprocess_config);
    
    ind_models(ss)=load_individual_model(trf_config);
    if trf_config.separate_conditions
        fprintf('TODO: this is just a place holder...\n')
        % will just have to index each model struct before feeding into
        % plot_model_weights
    else
        plot_model_weights(ind_models(ss),trf_config,plot_chns)
    end
end

function plot_model_weights(model,trf_config,chns)
    arguments
        model (1,1) struct
        trf_config (1,1) struct
        chns = 'all'
    end
    %TODO: use a switch-case to handle plotting a particular channel, the
    %best channel, or all channels...
    subj=trf_config.subj;
    title_str=sprintf('subj %d - %s chns',subj,chns);
    figure
    mTRFplot(model,'trf','all',chns);
    title(title_str)
end
function model=load_individual_model(trf_config)
    disp(['TODO: fix bug - this function is return trf_config ...' ...
        ' also will be a problem in analysis script'])
    if exist(trf_config.trf_model_path,'file')
        model_checkpoint=load_checkpoint(trf_config.trf_model_path,trf_config);
        if ismember('model',fieldnames(model_checkpoint))
            model=model_checkpoint.model;
        else
            fprintf('no model found in %s\n',trf_config.trf_model_path)
        end
    end
end

