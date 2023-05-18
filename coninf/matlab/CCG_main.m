function CCG_main(file_path, duration)

    
    addpath(genpath('../matlab')); % path to buzlab scripts & npy-matlab scripts
    
    
    ccg_params.binSize = .001; %1ms
    ccg_params.duration = 0.05; %50ms 
    ccg_params.epoch = [0 inf]; %whole session
    ccg_params.conv_w = .010/ccg_params.binSize;  % results in 5ms sigma (gaussian) 
    ccg_params.alpha = 0.01; %high frequency cut off, must be .001 for causal p-value matrix, not used in the current script.

    display(file_path); % location where spike trains are stored (in .npy format)
    display(['computing duration ' duration]); % 'duration' sets the duration of recording/simulation we want to use.
    
    
    file_path  = char(file_path);
    
    %load files 
    spktimes = readNPY([file_path '/spktrain_ms.npy']); % in milliseconds.
    spktmps = readNPY([file_path '/spktmps.npy']);

    
    bool_idx = spktimes<(str2double(duration)*1000);
    
    spktimes = spktimes(bool_idx);
    spktmps = spktmps(bool_idx);

    %store original python indices
    
    py_idx = unique(spktmps);
    new_spktmps = spktmps;
    %ccg c script doesn't accept discontinous indices
    for ii=1:length(py_idx)
       idx = ismember(spktmps, py_idx(ii));
       new_spktmps(idx) = ii; 
      
    end
    
    % initialize parpool first to make sure that it doesn't get count when measuring computation time.
    if(isempty(gcp('nocreate')))
        pool = parpool('local');
    end
    
    % start measuring computing time 
    tic;
    [Pvals, types]=infer_CCG(double(spktimes)/1000, new_spktmps, ccg_params); % times are in seconds.
    T = toc;
    
    
    Pvals_real = real(Pvals);
    
        
    % convert string into char vec
    duration = char(duration);

    save([file_path '/ccg_result_' duration], 'Pvals_real', 'types', 'T', 'py_idx')
    display(['done :' [file_path '/ccg_result_' duration]]);
    delete(pool);
   
    
end