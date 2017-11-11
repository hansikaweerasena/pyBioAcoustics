function [ vectorout, out ] = segmentation2( shortAvg, longAvg, signal, threshold )
%SEGMENTATION2 Summary of this function goes here
%   Detailed explanation goes here

l_vectorin = length(shortAvg);
set = 0;
segments = 0;
out = zeros(1,l_vectorin);
longAvg = threshold * longAvg;

vectorout = {};
temp = [];


for i=1:l_vectorin
    if shortAvg(i) > longAvg(i)
        temp = cat(1,temp,signal(i));
        set = 0;
        %out(i) = 1;
        
    elseif set == 0
        if length(temp) > 3000
            vectorout = cat(1,vectorout, temp);
            for j=i-length(temp):i-1
                out(j) = 1;
            end
            segments = segments + 1;
        end
        temp = [];
        set = 1;
    end
end