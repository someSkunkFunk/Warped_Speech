function checkpoint_data=load_checkpoint(load_path,expected_config)
% my custom loading function for loading saved checkpoint vars while
% validating configs
disp('this does nothing special yet.')
checkpoint_data=load(load_path);
config_fieldname=get_config_fieldname(checkpoint_data);
load_config=checkpoint_data.(config_fieldname{:});
mismatched_fields=validate_configs(expected_config,load_config);
if all_are_paths(mismatched_fields)
    fprintf(['since not checking fields that are uncommon ' ...
        'to both expected and loaded config, this will ' ...
        'likely erroneously return validated=true when ' ...
        'fieldnames are updated in case where expected config ' ...
        'has changed fields compared to previous version....\n']);
    % replace checkpoint data config and save it to file also
    checkpoint_data.(config_fieldname{:})=expected_config;    
    % create temporary struct to save updated config to file without
    % overwriting other variables in the file
    temp_data.(config_fieldname{:})=expected_config;
    % update_saved_config_paths(load_path,expected_config);
    save(load_path,'-struct','temp_data',config_fieldname{:},'-append')
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
        % validated=false;
        % (# fields, value_expected,value_load, fieldname)
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
        paths_only=true;
        for ii_field=1:length(mismatched_fields)
            % skip empty ones... first nonpath returns false
            if ~isempty(mismatched_fields{ii_field})
                expected_val=mismatched_fields{ii_field,1};
                field_name=mismatched_fields{ii_field,3};
                if contains(field_name,'config')
                    fprintf('%s config field not replaced but also not registering as erroneous.\n')
                elseif ~(isfolder(expected_val)||isfile(expected_val))
                        paths_only=false;
                end
            end
        end
    end
    function update_saved_config_paths(load_path,expected_config)
        %overwrite existing config if only difference is paths
        ;
    end
end