function [vr,out] = getVoicedClusterRegions(thresh_sound,tolerance)  
    out = thresh_sound;
    i = 1;
    found_i = 1;
    found_j = 1;
    
    status = 0;
    
    voiced_regions = {};
    logic_high = 1;
    while i <= length(thresh_sound)
        if(thresh_sound(i) == logic_high && ~status)
            found_i = i;
            status = 1;
        elseif(thresh_sound(i) == logic_high && status)
            found_j = i;
%           status = 1;
        else
            if(status)
                lb = found_i-tolerance;
                ub = found_j+tolerance;
                if((found_i-tolerance)< 0) 
                    lb = 1;
                end 
                if((found_j+tolerance)> length(thresh_sound))
                    ub = length(thresh_sound); 
                end
                n = lb:ub;
                out(lb:ub) = logic_high*ones(1,length(n));
                
                voiced_regions = cat(2,voiced_regions,n);
                
                found_i = 1;
                found_j = 1;
            end
            status = 0 ;
        end
        i = i + 1;
        
    end
    
    if(status)
        lb = found_i-tolerance;
        ub = length(thresh_sound);
        if((found_i-tolerance)< 0) 
        	lb = 1;
        end 
        n = lb:ub;
        out(lb:ub) = logic_high*ones(1,length(n));
                
        voiced_regions = cat(2,voiced_regions,n);
                
     end
     status = 0 ;
    
vr = voiced_regions;

