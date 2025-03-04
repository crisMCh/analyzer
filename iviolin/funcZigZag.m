function OutMat = funcZigZag(InMat)
%% Function summary

% Definitions: 
% Convert an [r,c] size matrix into a vector following a ZigZag path (1st
% row down, 2nd row up, 3rd row down, 4th row up, ....)


[r,c]=size(InMat);
OutMat=zeros(r*c,1);
for m=1:c
    if mod(m,2) == 0 %Even Columns
        OutMat((m-1)*r+1:(m-1)*r+r,1)=flipud(InMat(:,m));
    else % Odd colums
        OutMat((m-1)*r+1:(m-1)*r+r,1)=InMat(:,m);    
    end %if
end %for

end % End of this function