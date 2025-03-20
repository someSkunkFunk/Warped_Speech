function checkpoint_data=checkpoint_load(load_path,expected_config)
% my custom loading function for loading saved checkpoint vars while
% validating configs
disp('this does nothing special yet.')
checkpoint_data=load(load_path);
config_fieldname=get_config_fieldname(checkpoint_data);
load_config=checkpoint_data.(config_fieldname{:});
[validated,fields_diff]=validate_configs(expected_config,load_config);
% if validated&&all_diffs_paths(fields_diff)
% if validated&&all()


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
    function [validated,fields_diff]=validate_configs(expected_config, ...
            load_config)
        validated=false;
        % (# fields, value_expected,value_load, fieldname)
        fields_diff=cell(max(numel(fieldnames(expected_config)), ...
            numel(fieldnames(load_config))),3);
        if isequal(expected_config,load_config)
            validated=true;
        else
            %get fieldnames and values out, inpsect if only difference is path-based
            %TODO: consider case where field names in expected vs saved
            %config have changed (probably want to use setdiff here)
            
            expected_fields=fieldnames(expected_config);
            load_fields=fieldnames(load_config);
            common_fields=intersect(expected_fields,load_fields);
            n_diff_fields=0;
            for fldnm=common_fields
                expected_val=expected_config.fldnm;
                load_val=load_config.fldnm;
                if ~isequal(expected_val,load_val)
                    n_diff_fields=n_diff_fields+1;
                    fields_diff(n_diff_fields,:)={expected_val,load_val,fldnm};
                end
            end
            if all_diffs_paths(fields_diff)
                fprintf(['since not checking fields that are uncommon ' ...
                    'to both expected and loaded config, this will ' ...
                    'likely erroneously return validated=true when ' ...
                    'fieldnames are updated in case where expected config ' ...
                    'has changed fields compared to previous version....']);
                validated=true;
            end
        end
    end
    function update_saved_config(load_path,expected_config)
        %overwrite existing config if only difference is paths
    end
end