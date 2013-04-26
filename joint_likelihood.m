function [ nll, dnll ] = ...
    joint_likelihood( combined_params, Y, example_params_struct, censoring_func, num_censoring_samples )
    % Computes the joint likelihood of the whole model.
    
    % First, unpack parameters.  
    cur_params = rewrap( example_params_struct, combined_params );

    [ nll_latent, dnll_latent_X ] = ...
        latent_likelihood( cur_params.X );
    %checkgrad('latent_likelihood', cur_params.X, 1e-6 );
    
    % Correction factor to account for missing data.
    nll_censoring  = -censoring_likelihood( cur_params.X, Y, cur_params.log_hypers, censoring_func, num_censoring_samples);
    nll_latent = nll_latent - nll_censoring;
    
    [ nll_lvm, dnll_lvm_X, dnll_log_hypers ] = ...
        gplvm_likelihood( cur_params.X, Y, cur_params.log_hypers );
   % checkgrad('gplvm_likelihood', combined_params, 1e-6);
    
    nll =  nll_lvm + nll_latent;
    
    % Put gradients back into a vector.
    all_grads_struct.X = dnll_lvm_X + dnll_latent_X;
    all_grads_struct.log_hypers = dnll_log_hypers;
    dnll = unwrap( all_grads_struct );
end
