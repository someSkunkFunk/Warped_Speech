function checkpoint_data=load_checkpoint(load_path,expected_config)
%TODO: if loading TRF_config... check if at separate conditions step, in
%which case gotta either save configs separately and ignore the one in file
%already OR append to existing struct and only check that second struct
%fields match (except for maybe paths)
desired_config_found=false;
try
% my custom loading function for loading saved checkpoint vars while
% validating configs
    checkpoint_data=load(load_path);
    config_fieldname=get_config_fieldname(checkpoint_data);
    load_config=checkpoint_data.(config_fieldname{:});
    n_configs=numel(load_config);
    for nc=1:n_configs
    
        mismatched_fields=validate_configs(expected_config,load_config(nc));
        if all_are_paths(mismatched_fields)
            fprintf(['since not checking fields that are uncommon ' ...
                'to both expected and loaded config, this will ' ...
                'likely erroneously return validated=true when ' ...
                'fieldnames are updated in case where expected config ' ...
                'has changed fields compared to previous version....\n']);
            
            desired_config_found=true;
            % create temporary struct to save updated config to file without
            % overwriting other variables in the file
            temp_data.(config_fieldname{:})=expected_config;
            % update_saved_config_paths(load_path,expected_config);
            save(load_path,'-struct','temp_data',config_fieldname{:},'-append')
            % exit loop
            break
        % elseif 
        end
    end
    if desired_config_found
        % replace checkpoint data config (which may contain multiple) with
        % single desired config
            checkpoint_data.(config_fieldname{:})=expected_config;
    else
        error('specified config not found in file: %s',load_path);
    end

catch ME
    fprintf('wtf...')
    rethrow(ME)
end

    function config_fieldname=get_config_fieldname(checkpoint_data)
        %NOTE function assumes (for simplicity) that only ony config file
        %is contained....
        %TODO: need to update such that it also finds configs contained as
        %subfields of outer-config
        checkpoint_fieldnames=fieldnames(checkpoint_data);
        config_fieldmask=cellfun(@(x) contains(x,'config'),checkpoint_fieldnames);
        config_fieldname=checkpoint_fieldnames(config_fieldmask);
        if numel(config_fieldname)~=1
            fprintf('number of config-matched vars in checkpoint file:%d\n', ...
                numel(config_fieldname))
            error('either there is no config struct in checkpoint mat file or there are too many')
        end   
    end
    function mismatched_fields=validate_configs(expected_config, ...
            load_config)
        %% STUFF BELOW ASSUMES SINGLE CONFIG GIVEN
        % note that preallocating will leave some empty arrays at end...
        mismatched_fields=cell(max(numel(fieldnames(expected_config)), ...
            numel(fieldnames(load_config))),3);
        if isequal(expected_config,load_config)
            validated=true;
        else
            %get fieldnames and values out, inpsect if only difference is path-based
            %TODO: consider case where field names in expected vs saved
            %config have changed (probably want to use setdiff here)
            
            expected_fields=fieldnames(expected_config);
            load_fields=fieldnames(load_config);
            % look only at fields that have the same name between the two
            % structs
            common_fields=intersect(expected_fields,load_fields);
            n_mm_fields=0;
            for field_ii=1:length(common_fields)
                fldnm=common_fields{field_ii};
                expected_val=expected_config.(fldnm);
                load_val=load_config.(fldnm);
                if ~isequal(expected_val,load_val)
                    n_mm_fields=n_mm_fields+1;
                    mismatched_fields(n_mm_fields,:)={expected_val,load_val,fldnm};
                end
            end
            
        end
    end
    function paths_only=all_are_paths(mismatched_fields)
        %TODO: smarter way of doing this??? only needs to run once probably
        %though..
        %TODO: 
        try
            paths_only=true;
            for ii_field=1:length(mismatched_fields)
                % skip empty ones... first nonpath returns false
                if ~isempty(mismatched_fields{ii_field})
                    expected_val=mismatched_fields{ii_field,1};
                    field_name=mismatched_fields{ii_field,3};
                    %NOTE: I think checking config first below is necessary
                    %so that isfolder/isfile check doesnt cause error on
                    %config... i think we want the string check to go
                    %second to avoid additional error if any paths are not
                    %strings
                    switch(ischar(expected_val)||isstring(expected_val))
                        case 1
                            if ~(isfolder(expected_val)||isfile(expected_val))
                                paths_only=false;
                                break
                            end
                        case 0
                            if contains(field_name,'config')
                                fprintf('%s config field path values not replaced but also not registering as erroneous.\n')
                            else
                                paths_only=false;
                                break
                            end
                    end
                end
            end
        catch ME
            fprintf('something is not a string here...?\n%s\n',expected_val)
            rethrow(ME)
        end
    end

end
