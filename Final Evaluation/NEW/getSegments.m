function [ segments ] = getSegments( readFile )
%GETSEGMENTS is a function to do segmentation for audio file.
%   inputs: bird - name of the bird
%   inputs: recordings - number of recordings available for that bird
%   outputs: segments

    [roi,Fs] = audioread(readFile);
    
    % Local Processing
    
    Wn = 2000/Fs;
    [b, a] = butter(10, Wn, 'high');
    roi = (filter(b,a,roi))';

    %N = length(roi); % signal length
    %n = 0:N-1;

    %{
    wintype = 'rectwin';
    winlen = 101;
    winamp = 0.5*(1/winlen);

    out = (winlen-1)/2:(N+winlen-1)-(winlen-1)/2;
    out = out(1:N);

    [voiced_components, ~] = VoiceFilteringSTE(signal,wintype,winlen,winamp,out,Fs);
    clear signal out;
    disp(strcat('Number of ROI = ',num2str(length(voiced_components))));
%}
   % for region = 1: length(voiced_components)

        %disp(strcat('Processing region ',num2str(region)));

        % Apply spectral substraction
    %roi = voiced_components{region};
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

        if(r==0)
            energyST = cat(2,energyST,sum((diag(sparse(wHamm)) * (buffer(roi(s:e), winLen, winOverlap, 'nodelay'))).^2,1));
        else
            energyST = cat(2,energyST,sum((diag(sparse(wHamm)) * (buffer(roi(s-150:e), winLen, winOverlap, 'nodelay'))).^2,1));
        end
    end
    %energyST = sum((diag(sparse(wHamm)) * (buffer(roi, winLen, winOverlap, 'nodelay'))).^2,1);
    energyST = cat(2,zeros(1,150),energyST);
    energyST = cat(2,energyST,zeros(1,150));

    % Average energy
    shortAvg = moving_avg(energyST,1024,512);

    clear energyST;

    % Break into segments
    [segments] = segmentation(shortAvg,roi,.1*mean(shortAvg));
    clear roi shortAvg;

    %disp(strcat('Number of segments = ',num2str(length(segments))));

    for i = 1:length(segments)
        if Fs ~= 44100
            [P,Q] = rat(44.1e3/Fs);
            segments{i} = resample(segments{i},P,Q);
        end
    end

%    end

end