
% assumes plot_trfs has been called to generate ind_models
close all
cc_ii_=3;
ss_ii_=5;
% plot distribution of model weights
figure
histogram(ind_models(ss_ii_,cc_ii_).w(:)), title(sprintf('weights, subj=%d, cond=%d',ss_ii_,cc_ii_))
% plot distribution of model intercept terms
figure
histogram(ind_models(ss_ii_,cc_ii_).b(:)), title(sprintf('intercepts, subj=%d, cond=%d',ss_ii_,cc_ii_))
clear cc_ii_ ss_ii_