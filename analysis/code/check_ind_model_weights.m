function check_ind_model_weights(ind_models)
if size(ind_models,2)==3
    for si=1:n_subjs, disp([si, size(ind_models(si,1).w),size(ind_models(si,2).w),size(ind_models(si,3).w)]), end
elseif size(ind_models,2)==1
    for si=1:n_subjs, disp([si, size(ind_models(si).w)]),end
end
end