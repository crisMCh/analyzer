function S = funcFindLargestSquares(I)
%% Function summary
% Find largest squares: finds largest sqare regions with all points set to 1.

% Function Inputs:

% Function Returns:
%   

% Updates:
% 

%% Search for the largest connected squares
[nr nc] = size(I);
S = double(I>0);
for r=(nr-1):-1:1
  for c=(nc-1):-1:1
    if (S(r,c))
      a = S(r  ,c+1);
      b = S(r+1,c  );
      d = S(r+1,c+1);
      S(r,c) = min([a b d]) + 1;
    end
  end  
end

%End of this function