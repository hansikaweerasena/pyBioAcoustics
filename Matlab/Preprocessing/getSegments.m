function [ segments ] = getSegments( bird, recordings )
%GETSEGMENTS is a function to do segmentation for audio file.
%   inputs: bird - name of the bird
%   inputs: recordings - number of recordings available for that bird
%   outputs: segments

    segmentNumber = 0;

    for recording = 1:recordings

        % Read audio file
        readFile = strcat(bird,num2str(recording),'.wav');
        [signal,Fs] = audioread(readFile);
        
        disp(strcat('Processing file: ',readFile));
                
        % Local Processing
        Wn = 2000/Fs;
        [b, a] = butter(10, Wn, 'high');
        signal = (filter(b,a,signal))';

        N = length(signal); % signal length
        %n = 0:N-1;

        wintype = 'rectwin';
        winlen = 101;
        winamp = 0.5*(1/winlen);

        out = (winlen-1)/2:(N+winlen-1)-(winlen-1)/2;
        out = out(1:N);
        t = (out-(winlen-1)/2)*(1/Fs);
        
        [voiced_components, ~] = VoiceFilteringSTE(signal,t,wintype,winlen,winamp,out,Fs);
        clear signal out t;
        disp(strcat('Number of ROI = ',num2str(length(voiced_components))));
        
        for region = 1: length(voiced_components)
            
            disp(strcat('Processing region ',num2str(region)));
            
            % Apply spectral substraction
            roi = voiced_components{region};
            roi = SSBoll79(roi,Fs,0.5);

            % A hamming window is chosen
            winLen = 301;
            winOverlap = 300;
            wHamm = hamming(winLen);

            % Framing and windowing the signal without for loops.
            %sigFramed = buffer(roi, winLen, winOverlap, 'nodelay');
            %sigWindowed = diag(sparse(wHamm)) * sigFramed;

            % Short-Time Energy calculation
            energyST = [];
            for r=0:floor(length(roi)/831000)
                s = r*831000+1;
                e = min(s+830999+150,length(roi));
                energyST = cat(2,energyST,sum((diag(sparse(wHamm)) * (buffer(roi(s:e), winLen, winOverlap, 'nodelay'))).^2,1));
            end
            %energyST = sum((diag(sparse(wHamm)) * (buffer(roi, winLen, winOverlap, 'nodelay'))).^2,1);
            energyST = cat(2,zeros(1,150),energyST);
            % Average energy
            shortAvg = moving_avg2(energyST,1024,512);
            clear energyST;

            % Break into segments
            [segments] = segmentation(shortAvg,roi,.1*mean(shortAvg));
            clear roi shortAvg;
            
            disp(strcat('Number of segments = ',num2str(length(segments))));

            for i = 1:length(segments)
                segmentName = strcat(bird,'_segment',num2str(segmentNumber+i),'.wav');
                %segmentName = strcat(file,'_segment1.wav');
                %segmentName = strcat(segmentName,'.wav');
                if Fs ~= 44100
                    [P,Q] = rat(44.1e3/Fs);
                    segments{i} = resample(segments{i},P,Q);
                end
                audiowrite(segmentName,segments{i},44100);
            end

            segmentNumber = segmentNumber + length(segments);

            clear readFile segments

        end
        
        clear voiced_components signal
    end
end