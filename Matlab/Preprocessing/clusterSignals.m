function y = clusterSignals(in_sound,checking_rng)  

    out_sound = in_sound;
    status = 0;

    j = 1;
    i = 1;
    while i <= length(in_sound)

        if(in_sound(i) == 1)

            while( j <= checking_rng && ((i+j) <= length(in_sound)))
                if(in_sound(i+j) == 1)
                   status = 1;
                   break;
                end
                j = j + 1;
                
                if i+j > length(in_sound)
                   break; 
                end   
            end

            if(status)
                while j > 1
                    out_sound(i+j-1) = 1;
                    j = j -1;
                end
            end

            status = 0;
            i = i + j;
            j = 1;
        else
            i = i + 1;
        end


        
    end
y =  out_sound;
