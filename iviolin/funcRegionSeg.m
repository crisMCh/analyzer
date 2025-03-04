function seg = funcRegionSeg(I,init_mask,max_its,alpha,display)
% 03.02.2018

% Region Based Active Contour Segmentation
% Implementation based on the paper "Active Contours Without
% Edges" (Chan Vese). Coded by: Shawn Lankton (www.shawnlankton.com)

% seg = region_seg(I,init_mask,max_its,alpha,display)
%
% Inputs: I           2D image
%         init_mask   Initialization (1 = foreground, 0 = bg)
%         max_its     Number of iterations to run segmentation for
%         alpha       (optional) Weight of smoothing term
%                       higer = smoother.  default = 0.2
%         display     (optional) displays intermediate outputs
%                       default = true
%
% Outputs: seg        Final segmentation mask (1=fg, 0=bg)

  %Check if another alpha is provided. The default value is 0.2
  if(~exist('alpha','var')) 
    alpha = .2; 
  end
  
  %Check how to display some intermediate results for debugging
  if(~exist('display','var'))
    display = true;
  end
  
  % Make I 2D double matrix
  I = im2graydouble(I);    
  
  % Create a signed distance map (SDF) from mask
  phi = mask2phi(init_mask);
  
  % This is the main loop
  for its = 1:max_its   % Note: no automatic convergence test

    idx = find(phi <= 1.2 & phi >= -1.2);  %get the curve's narrow band
    
    % Find the interior and exterior mean values
    upts = find(phi<=0);                 % interior points
    vpts = find(phi>0);                  % exterior points
    u = sum(I(upts))/(length(upts)+eps); % interior mean
    v = sum(I(vpts))/(length(vpts)+eps); % exterior mean
    
    F = (I(idx)-u).^2-(I(idx)-v).^2;         % force from image information
    curvature = get_curvature(phi,idx);  % force from curvature penalty
    
    dphidt = F./max(abs(F)) + alpha*curvature;  % gradient descent to minimize energy
    
    % Maintain the CFL condition
    dt = .45/(max(dphidt)+eps);
        
    % Evolve the curve
    phi(idx) = phi(idx) + dt.*dphidt;

    % Keep SDF smooth
    phi = sussman(phi, .5);

    % Show intermediate output
    if((display>0)&&(mod(its,20) == 0)) 
      showCurveAndPhi(I,phi,its);  
    end
  end
  
  % Show the final output
  if(display)
    showCurveAndPhi(I,phi,its);
  end
  
  % Generate a mask from SDF
  seg = phi<=0; %-- Get mask from levelset

  
%---------------------------------------------------------------------
%Additional functions:
  
  
% Display the image with the superimposed curve
function showCurveAndPhi(I, phi, i)
  imshow(I,'initialmagnification',200,'displayrange',[0 255]); hold on;
  contour(phi, [0 0], 'g','LineWidth',4);
  contour(phi, [0 0], 'k','LineWidth',2);
  hold off; title([num2str(i) ' Iterations']); drawnow;
  
% Convert mask to a SDF
function phi = mask2phi(init_a)
  phi=bwdist(init_a)-bwdist(1-init_a)+im2double(init_a)-.5;
  
% Compute the curvature along SDF
function curvature = get_curvature(phi,idx)
    [dimy, dimx] = size(phi);        
    [y x] = ind2sub([dimy,dimx],idx);  % get subscripts

    % Get subscripts of neighbors
    ym1 = y-1; xm1 = x-1; yp1 = y+1; xp1 = x+1;

    % Bounds checking  
    ym1(ym1<1) = 1; xm1(xm1<1) = 1;              
    yp1(yp1>dimy)=dimy; xp1(xp1>dimx) = dimx;    

    % Get indexes for 8 neighbors
    idup = sub2ind(size(phi),yp1,x);    
    iddn = sub2ind(size(phi),ym1,x);
    idlt = sub2ind(size(phi),y,xm1);
    idrt = sub2ind(size(phi),y,xp1);
    idul = sub2ind(size(phi),yp1,xm1);
    idur = sub2ind(size(phi),yp1,xp1);
    iddl = sub2ind(size(phi),ym1,xm1);
    iddr = sub2ind(size(phi),ym1,xp1);
    
    % Get central derivatives of SDF at x,y
    phi_x  = -phi(idlt)+phi(idrt);
    phi_y  = -phi(iddn)+phi(idup);
    phi_xx = phi(idlt)-2*phi(idx)+phi(idrt);
    phi_yy = phi(iddn)-2*phi(idx)+phi(idup);
    phi_xy = -0.25*phi(iddl)-0.25*phi(idur)...
             +0.25*phi(iddr)+0.25*phi(idul);
    phi_x2 = phi_x.^2;
    phi_y2 = phi_y.^2;
    
    % Compute curvature (Kappa)
    curvature = ((phi_x2.*phi_yy + phi_y2.*phi_xx - 2*phi_x.*phi_y.*phi_xy)./...
              (phi_x2 + phi_y2 +eps).^(3/2)).*(phi_x2 + phi_y2).^(1/2);        
  
% Converts image to one channel (grayscale) double
function img = im2graydouble(img)    
  [dimy, dimx, c] = size(img);
  if(isfloat(img)) % image is a double
    if(c==3) 
      img = rgb2gray(uint8(img)); 
    end
  else           % image is a int
    if(c==3) 
      img = rgb2gray(img); 
    end
    img = double(img);
  end

% Level set re-initialization by the Sussman method
function D = sussman(D, dt)
  % Forward/backward differences
  a = D - shiftR(D); % backward
  b = shiftL(D) - D; % forward
  c = D - shiftD(D); % backward
  d = shiftU(D) - D; % forward
  
  a_p = a;  a_n = a; % a+ and a-
  b_p = b;  b_n = b;
  c_p = c;  c_n = c;
  d_p = d;  d_n = d;
  
  a_p(a < 0) = 0;
  a_n(a > 0) = 0;
  b_p(b < 0) = 0;
  b_n(b > 0) = 0;
  c_p(c < 0) = 0;
  c_n(c > 0) = 0;
  d_p(d < 0) = 0;
  d_n(d > 0) = 0;
  
  dD = zeros(size(D));
  D_neg_ind = find(D < 0);
  D_pos_ind = find(D > 0);
  dD(D_pos_ind) = sqrt(max(a_p(D_pos_ind).^2, b_n(D_pos_ind).^2) ...
                       + max(c_p(D_pos_ind).^2, d_n(D_pos_ind).^2)) - 1;
  dD(D_neg_ind) = sqrt(max(a_n(D_neg_ind).^2, b_p(D_neg_ind).^2) ...
                       + max(c_n(D_neg_ind).^2, d_p(D_neg_ind).^2)) - 1;
  
  D = D - dt .* sussman_sign(D) .* dD;
  
% Whole matrix derivatives
function shift = shiftD(M)
  shift = shiftR(M')';

function shift = shiftL(M)
  shift = [ M(:,2:size(M,2)) M(:,size(M,2)) ];

function shift = shiftR(M)
  shift = [ M(:,1) M(:,1:size(M,2)-1) ];

function shift = shiftU(M)
  shift = shiftL(M')';
  
function S = sussman_sign(D)
  S = D ./ sqrt(D.^2 + 1);    

  



