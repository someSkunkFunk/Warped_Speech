function match=configs_match(data_path,config_to_match)
    % helper for checking if matching config is contained in saved
    % data_path
    match=false;
    data_vars=who('-file',data_path);
    config_mask=cellfun(@(x) contains(x,'config'), data_vars);
    % note - might have to change line below when multiple configs
    % contained in file
    if sum(config_mask)>1
        disp('double-check this approach')
    end
    % load the configs from file
    load(data_path,data_vars{config_mask})
    if exist('trf_config','var')&&exist('preprocess_config','var')
        error('assuming only one of these is in storage!')
    end
    if exist('preprocess_config','var')
        pre_matches=false(numel(preprocess_config),1);
        preprocess_params_to_check={'bpfilter','ref','fs',...
            'subj','bad_chans'};
        for nn=1:numel(preprocess_config)
            for pp=1:numel(preprocess_params_to_check)
                pp_field=preprocess_params_to_check{pp};
                if ~isequal(preprocess_config.(pp_field),config_to_match.(pp_field))
                    break
                end
                % if it doesnt break on last var -> all are equal
                if pp==numel(preprocess_params_to_check)
                    pre_matches(nn)=true;
                end
            end
        end
        if any(pre_matches)
            match=true;
        end
    end
    
    %note: if matching trf configs, there should be a sub-structure array
    %with preprocessing params we can also check for equality, but no
    %standalone preprocess_config array
    if exist('trf_config','var')
        %todo
        error('this does nothing yet')
    end
end
