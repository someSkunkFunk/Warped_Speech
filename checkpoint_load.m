function checkpoint_data=checkpoint_load(checkpoint_mat_path,expected_config)
% my custom loading function for loading saved checkpoint vars while
% validating configs

%TODO: validate that configs match except perhaps for path-based variables,
%in which case we can just update them to reflect what the new config
%specifies
disp('this does nothing special yet')
checkpoint_data=load(load_path);

%TODO: account for case where the config to edit is contained within a
%field of another config (such as preprocess_config within trf_config)
end