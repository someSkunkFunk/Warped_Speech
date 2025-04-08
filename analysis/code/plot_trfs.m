clearvars -except user_profile boxdir_mine boxdir_lab
%%
%plotting params

%%
subjs=2:17;
plot_chns=85;
separate_conditions=true; %NOTE: not operator required when 
    % initializing config_trf since technically it expects 
    % "do_lambda_optimization" as argument 
    % ignoring the false case rn sincee buggy but not a priority but should
    % fix
n_subjs=numel(subjs);
plot_individual_weights=true;
plot_avg_weights=true;
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
        % if trf_config.separate_conditions
        %     % fprintf('p\n')
        %     % will just have to index each model struct before feeding into
        %     % plot_model_weights
        %     for cc=1:size(ind_models,2)
        %         plot_model_weights(ind_models(ss,cc),trf_config,plot_chns)
        %     end
        % else
        %     plot_model_weights(ind_models(ss,1),trf_config,plot_chns)
        % end
    % end
end
if plot_avg_weights
    plot_model_weights(ind_models,trf_config,plot_chns)
end

function plot_model_weights(ind_models,trf_config,chns)
%TODO: use a switch-case to handle plotting a particular channel, the
        %best channel, or all channels... needs to chance title string
        %format indicator
    arguments
        ind_models struct
        trf_config (1,1) struct
        chns = 'all'
    end
    n_subjs=size(ind_models,1);
    if trf_config.separate_conditions
        n_conditions=size(ind_models,2);
        fprintf('plotting condition-specific TRFs...\n ')
    else
        n_conditions=1;
        fprintf("NOTE: line above this will bypass bad config handling in " + ...
            "separate_conditions=false case but that also means the issue " + ...
            "will be hidden and still occur so needs to be fixed directly...\n")
    end
    conditions={'fast','og','slow'};
    for cc=1:n_conditions
        if n_subjs==1
            fprintf('plotting individual model weights...\n')
            %plot individual subject weights            
            title_str=sprintf('subj: %d - chns %s - %s',trf_config.subj, ...
                num2str(chns),conditions{cc});
            figure
            mTRFplot(ind_models(1,cc),'trf','all',chns);
            title(title_str)
        else
            % assume we want to average out the weights and plot them
            % compile weights into single model struct for current
            % condition
            fprintf('plotting subject-averaged model weights...\n')
            avg_model=construct_avg_model(ind_models);

            % NOTE: repetitive code below could be consolidated across
            % single condition vs separate condition trf cases...
            title_str=sprintf('avg TRF - chns: %s - condition: %s ', ...
                num2str(chns),conditions{cc});
            figure
            mTRFplot(avg_model,'trf','all',chns);
            title(title_str)
            
        
        end
    end
end

function avg_models=construct_avg_models(ind_models)
    [~,n_weights,n_chans]=size(model(1,1).w);
    W_stack=nan(n_subjs,n_weights,n_chans);
    avg_model=struct();
    model_fields=fieldnames(model(1,cc));
    for ss=1:n_subjs
        W_stack(ss,:,:)=model(ss,cc).w;
    end

    fprintf(['NOTE: avg model below will only have correct average weights',...
    'other fields which may vary at individual subject level could',...
    'be incorrect...\n'])

    for ff=1:numel(model_fields)
        ff_field=model_fields{ff};
        if strcmp(ff_field,'w')
            avg_model.(ff_field)=mean(W_stack,1);
        else
            avg_model.(ff_field)=model(1,cc).(ff_field);
        end
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

