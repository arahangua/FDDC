function [Pval,type]=infer_CCG(sorted_spk_vec, sorted_spk_idx, ccg_params)

%network inference CCG calculation script 

% necessary paths

addpath(genpath('../matlab')); % path to bz scripts & npy-matlab scripts


binSize = ccg_params.binSize; %1ms
duration = ccg_params.duration; %50ms
epoch = ccg_params.epoch; %whole session
conv_w = ccg_params.conv_w;  % 10ms window
alpha = ccg_params.alpha; %high frequency cut off, must be .001 for causal p-value matrix


indx_list=unique(sorted_spk_idx);
nCel=length(unique(sorted_spk_idx));

% Create CCGs (including autoCG) for all cells
[ccgR1,tR] = CCG(double(sorted_spk_vec),double(sorted_spk_idx),'binSize',binSize,'duration',duration);

ccgR = nan(size(ccgR1,1),nCel,nCel);
ccgR(:,1:size(ccgR1,2),1:size(ccgR1,2)) = ccgR1;


% get  CI for each CCG
Pval=nan(nCel,nCel);
type=zeros(nCel,nCel);


cell_pair = nchoosek(1:nCel,2);


result_cell = {};
parfor ii=1:size(cell_pair,1) 
    
    result=[]
    
    refcellID = cell_pair(ii,1);
    cell2ID = cell_pair(ii,2);

    %for cell2ID=1:max(nCel)
        
        if(refcellID==cell2ID)
           continue; 
        end
        
        cch=ccgR(:,refcellID,cell2ID);			% extract corresponding cross-correlation histogram vector
        

         % calculate predictions using Eran's bz_cch_conv
            
        [pvals,pred,qvals]=bz_cch_conv(cch,conv_w);
        
        % Find if significant periods falls in monosynaptic window +/- 5ms
        % (buffer +1)
        prebins = round(length(cch)/2 - .0050/binSize):round(length(cch)/2);
        postbins = round(length(cch)/2):round(length(cch)/2 + .005/binSize);

        % pre-post
        [v,w] = max(cch(postbins));
        [v,w_min]= min(cch(postbins));
 
        post_ex_pval = pvals(postbins(w));
        post_in_pval = 1-pvals(postbins(w_min));
        
        log_ex = -log(post_ex_pval);
        log_in = -log(post_in_pval);
        
        if(log_ex>=log_in)
           result.prepost_type=1;
           result.prepost_stat = log_ex;
        else
            result.prepost_type=-1;
            result.prepost_stat = log_in; 
        end
            
        
        
        % post-pre
        [v,w] = max(cch(prebins));
        [v,w_min]= min(cch(prebins));
        
        pre_ex_pval = pvals(prebins(w));
        pre_in_pval = 1-pvals(prebins(w_min));
        
        
        log_ex = -log(pre_ex_pval);
        log_in = -log(pre_in_pval);
        
        if(log_ex>=log_in)
           result.postpre_type=1;
           result.postpre_stat = log_ex;
        else
            result.postpre_type=-1;
            result.postpre_stat = log_in;
        end
        result_cell{ii} = result;
    
end



for jj =1:size(result_cell,2)
    
   curr = result_cell{jj};
   
   Pval(cell_pair(jj,1),cell_pair(jj,2)) = curr.prepost_stat;
   Pval(cell_pair(jj,2),cell_pair(jj,1)) = curr.postpre_stat;
    
   type(cell_pair(jj,1),cell_pair(jj,2)) = curr.prepost_type;
   type(cell_pair(jj,2),cell_pair(jj,1)) = curr.postpre_type;
    
end


end