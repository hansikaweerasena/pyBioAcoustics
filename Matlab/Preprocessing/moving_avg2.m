function [ out ] = moving_avg2( in, windowSize, overlap)
%MOVING_AVG Summary of this function goes here
%   Detailed explanation goes here
l_in = length(in);
iMax = l_in - windowSize;
noOverlap = windowSize - overlap;

wHann = hann(windowSize*2);
wHann = (wHann(windowSize:windowSize*2))';

%out = zeros(1,windowSize - overlap);
out = [];

avg = mean(in(1:1+windowSize) .* wHann);
%preAvg = mean(avg(1:windowSize - overlap));
out = cat(2,out,linspace(in(1),avg,windowSize - overlap));

%{
for j=1 : (windowSize - overlap)
    out(j) = mean(in(1:j));
end
%}

preAvg = out(windowSize - overlap);

i = windowSize - overlap + 1;

while i < iMax
    avg = mean(in(i:i+windowSize) .* wHann);
    temp = linspace(preAvg,avg,noOverlap);
    out = cat(2,out,temp);
    i = i + noOverlap;
    preAvg = avg;
    clear temp;
end


%i = i - noOverlap;

last = linspace(preAvg,in(l_in),l_in-i+1);
out = cat(2,out,last);


%{
for j=i+1 : l_in
    out(j) = mean(in(j:l_in));
end
%}
    
%{    
for i=1 : (l_in-windowSize)
    if i<=l_half_interval
        out(i)=mean(in(1:i));   % Then I calculate the average with values until the current value                                    % values until this one
    elseif i>l_half_interval&&i<(l_in-l_half_interval)
        out(i)=mean(in(i-l_half_interval:i+l_half_interval));
    elseif i>=(l_in-l_half_interval)
        out(i)=mean(in(i:l_in));   % Then I calculate the average with values until the current value                                    % values until this one
    end

%}

end

