function extract_from_path(path_local, target_neuron)
    splits = split(path_local,'_');
    brain_region = splits{1};
    brain_region = split(brain_region,'/');
    brain_region = brain_region{end};
    mouse_id = splits{2};
    date = splits{3};
    sess_id = splits{4};
    [dat, T, N, F, names, target_neuron_id,cumsum_explained] = GAM_Step1(path_local, target_neuron, 0.005); 
    new_file_name = sprintf('gam_preproc_neu%d_%s_%s_%s_%s',...
        target_neuron_id,brain_region,mouse_id,date,sess_id);
    try
        dat = rmfield(dat,'b');
    catch
        
    end
    try
        dat = rmfield(dat,'w');
    catch
    end
    try
        dat = rmfield(dat,'c');
    catch
    end
    try
        dat = rmfield(dat,'probe');
    catch
    end
    for kk  = {'stim', 'st', 'movement', 'metrics', 'feedback'}
        try
            dat.n = rmfield(dat.n,kk{1});
        catch
            continue
        end
    end
    
    save(new_file_name,'dat', 'T', 'N', 'F', 'names','cumsum_explained','-v6')
end