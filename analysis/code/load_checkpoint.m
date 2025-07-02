function checkpoint_data=load_checkpoint(load_path,expected_config,load_config_only)
%TODO: if loading TRF_config... check if at separate conditions step, in
%which case gotta either save configs separately and ignore the one in file
%already OR append to existing struct and only check that second struct
%fields match (except for maybe paths)
arguments
    load_path
    expected_config (1,1) struct
    load_config_only (1,1) logical = false;
end
found_desired_config=false;
overwrite_saved_config=false;
% load_config_only=false;
% error(['TODO: need to add option to bypass loading data vars when we just ' ...
%     'need the config for separate conditions case'])
% try
% my custom loading function for loading saved checkpoint vars while
% validating configs
checkpoint_data=load(load_path);
checkpoint_varnames=fieldnames(checkpoint_data);
config_mask=cellfun(@(x) contains(x,'config'),checkpoint_varnames);
config_fieldname=checkpoint_varnames(config_mask);
data_fieldnames=checkpoint_varnames(~config_mask);
% config_fieldname=get_config_fieldname(checkpoint_data);
load_config=checkpoint_data.(config_fieldname{:});
n_configs=numel(load_config);
for nc=1:n_configs
% loop thru different configurations
    mismatched_fields=validate_configs(expected_config,load_config(nc));
    if all(cellfun(@isempty,mismatched_fields),'all')
        found_desired_config=true;
        break
    elseif all_are_paths(mismatched_fields)
        %NOTE: all_are_paths will erroneously try and overwrite
        %existing config when mismatched_fields is empty

        fprintf(['since not checking fields that are uncommon ' ...
            'to both expected and loaded config, this will ' ...
            'likely erroneously return validated=true when ' ...
            'fieldnames are updated in case where expected config ' ...
            'has changed fields compared to previous version....\n']);
        
        found_desired_config=true;
        if overwrite_saved_config
            % create temporary struct to save updated config to file without
            % overwriting other variables in the file
            temp_data.(config_fieldname{:})=expected_config;
            % update_saved_config_paths(load_path,expected_config);
            % NOTE: muted line below was in case temp_data contained
            % multiple variables and we wanted to specifically save the one
            % corresponding to confing_fieldname... but I don't think we'll
            % end up with multple fields in temp_data so unnecessary...?
            % save(load_path,'-struct','temp_data',config_fieldname{:},'-append')
            fprintf(['need to update saving strategy so existing ' ...
                'config is not erased...?\n'])
            save(load_path,'-struct','temp_data','-append')
        end
        % exit loop
        break
    end
end


if found_desired_config
    % replace checkpoint data config (which may contain multiple) with
    % single desired config
    checkpoint_data.(config_fieldname{:})=checkpoint_data.(config_fieldname{:})(nc);
    if ~load_config_only
        for ff=1:numel(data_fieldnames)
            if strcmp(data_fieldnames{ff},'stats_null')
            %note: stats_null numbering is a problem below...
                checkpoint_data=rmfield(checkpoint_data,'stats_null');
                disp('ignoring stats_null in file.')
            else
                temp_data=checkpoint_data.(data_fieldnames{ff});
                %TODO: verify that triple dot indexing below works when struct
                %in data has less than 3 dimensions
                checkpoint_data.(data_fieldnames{ff})=temp_data(nc,:,:);
            end
        end
    end
else
    %NOTE: best_lam only needs to be copied from condition agnostic
    %trf_config when it is being evaluated for the first time, hence
    %why we should pull that up here (I think...?)
    % actually should put in analysis script cuz we'll need
    % preprocess_config to generate the associated
    % separate_condition=false config to pull from..
    fprintf('specified config not found in file: %s\n',load_path);
    checkpoint_data=struct();
end

    function mismatched_fields=validate_configs(expected_config, ...
            load_config)
        % assumes expected and load configs are both (1x1)
        % note that preallocating will leave some empty arrays at end...
        mismatched_fields=cell(max(numel(fieldnames(expected_config)), ...
            numel(fieldnames(load_config))),3);
        if ~isequal(expected_config,load_config)
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
        %though...
        try
            % NOTE ASSUMING TRUE BECAUSE ONLY FEEDING TO THIS FUNCTION WHEN
            % MISMATCHED FIELDS ARE NOT ALL EMPTY
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
                                return
                            end
                        case 0
                            if contains(field_name,'config')
                                fprintf(['%s config values not replaced but ' ...
                                    'also not registering as erroneous.\n'],field_name)
                            else
                                paths_only=false;
                                return
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
