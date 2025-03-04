function [ind,t0,s0] = funcZeroCrossing(S,t,level,imeth)
% Find the crossings of a given level of a signal

error(nargchk(1,4,nargin));
% Check the time vector input for consistency
if nargin < 2 || isempty(t)
	% If no time vector is given, use the index vector as time
    t = 1:length(S);
elseif length(t) ~= length(S)
	% If S and t are not of the same length, throw an error
    error('t and S must be of identical length!');    
end
% Check the level input
if nargin < 3
	% Set standard value 0, if level is not given
    level = 0;
end
% Check interpolation method input
if nargin < 4
    imeth = 'linear';
end

% Make 't' and 'S' row vectors
t = t(:)';
S = S(:)';

% Always search for zeros. For any other other threshold value "level", subtract it from
% the values and then search for zeros.
S   = S - level;
% First: look for exact zeros
ind0 = find( S == 0 ); 
% Second: look for zero crossings between data points
S1 = S(1:end-1) .* S(2:end);
ind1 = find( S1 < 0 );
% Bring exact zeros and "in-between" zeros together in one sorted array
ind = sort([ind0 ind1]);
% Pick the associated time values
t0 = t(ind); 
s0 = S(ind);
if strcmp(imeth,'linear')
    % Linear interpolation of crossing
    for ii=1:length(t0)
        if abs(S(ind(ii))) > eps*abs(S(ind(ii)))   
            
            % Interpolate only when data points are not already zero
            NUM = (t(ind(ii)+1) - t(ind(ii)));
            DEN = (S(ind(ii)+1) - S(ind(ii)));
            slope =  NUM / DEN;
            terme = S(ind(ii)) * slope;
            t0(ii) = t0(ii) - terme;
            % Simply set the value to zero: 
            s0(ii) = 0;
        end
    end
end
