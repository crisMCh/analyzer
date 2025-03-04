function NPSresult = funcCalcNPS(IMG,PatchSizeX,PatchSizeY,NPS_roi_size,NPS_pix_total,px,ImShowSizeX, ImShowSizeY, ROIname,PatientNumber,DEBUG_PLOT_LEVEL,ENV_TYPE,HOST_TYPE)
%% Function summary
% Date: 08.09.2020
% Compute the NPS for the provided ROI

% Definitions: 
% 

% Function Inputs:
%   ...
%   ImagePathInfo - Contains the complete image name and path info (used later for automatically save the results)
%   ROIname - Used as file name extension (eg '...-ROI1.jpg' or '...-ROI2.jpg'


% Function Returns:
%   

% Updates:



%% Additional parameter definition

%Size of figures for plotting
ImShowSizeX=400;ImShowSizeY=400; %Displayed image size using imshow


%% PATCH ANALYSIS (03.06.2020: now in function funcPatchAnalysis)
%For a squared shaped patch, just one analysis is performed
%For a rectangular shaped patch, 3 analyses are performed:
    %1: Rectangular patch as defined
    %2: Rectangular patch as defined and rotated by 90 degree
    %3: Squared patch with a length obtained from the average recangular
    %lengths
    %--> All three binary patterns are then combined using a logical AND
    %operator

if PatchSizeX == PatchSizeY %(if not square size)
    b1 = funcPatchAnalysis(IMG,PatchSizeX,PatchSizeY,DEBUG_PLOT_LEVEL,ENV_TYPE,PatientNumber);
    b = b1;
    bxSize = min(length(b(1,:)));
    bySize = min(length(b(:,1)));
else
    b1 = funcPatchAnalysis(IMG,PatchSizeX,PatchSizeY,PatientNumber);
    b2 = funcPatchAnalysis(IMG,PatchSizeY,PatchSizeX,PatientNumber); %Like b1, but rectangular patch was rotated
    SquPatchSize = floor(mean([PatchSizeX,PatchSizeY]));
    b3 = funcPatchAnalysis(IMG,SquPatchSize,SquPatchSize,PatientNumber); %Use a square patch with mean patch size of rect patch.

    bxSize = min([length(b1(1,:)),length(b2(1,:)),length(b3(1,:))]);
    bySize = min([length(b1(:,1)),length(b2(:,1)),length(b3(:,1))]);

    %Logical AND of all binary patterns obtained from the patch analysis
    b = b1(1:bySize,1:bxSize) & b2(1:bySize,1:bxSize) & b3(1:bySize,1:bxSize); 

end

%% NPS ROI Selection - Version 2: Find the largest region (largest squares) [PRODUCTION VERSION!]
NPS_roi_largest_squares = funcFindLargestSquares(b);
[max_num, max_idx]=max(NPS_roi_largest_squares(:));

% Update_ 04.02.2024: When there were not enough homogenous pixels for NPS
% estimation, set results all to zero and stop here.
if max_num < 4
    NPSpix_mean = 0;  
    %% Summarize results as a structure
    NPSresult.nps_measured = zeros(10,10);
    NPSresult.f =  zeros(10,10);
    NPSresult.nps_measured_radagv = zeros(1,10);
    NPSresult.f_avg = 0:0.1:0.9;
    NPSresult.var_measured = 0;
    NPSresult.var_NPS = 0;
    return
end

[Y,X]=ind2sub(size(NPS_roi_largest_squares),max_idx)
NPS_roi_size_max=NPS_roi_largest_squares(Y,X)
NPSpix = IMG(Y:Y+NPS_roi_size_max-1,X:X+NPS_roi_size_max-1);
if DEBUG_PLOT_LEVEL >= 2
    figure;imagesc(NPSpix);colormap(gray);title('Background pixel matrix for NPS computation (Version 2)')
end

%% NPS computation

%Deterend the pixel matrix
%NPSpix = detrend(NPSpix,2);
NPSpix = NPSpix - mean2(NPSpix);

NPS_roi_size = length(NPSpix(:,1));

%Measured variance
var_measured = var(NPSpix(:));
disp(['Pixel variance measured with var(): ' num2str((var_measured))])

%Compute the NPS
[nps_measured, f] = funcCalcDigitalNps(NPSpix, 2, px, 1, 0); % No NPS averaging
var_NPS = trapz(f, trapz(f, nps_measured)); % The variance is the integral of the NPS:
disp(['Pixel variance from measured NPS: ' num2str((var_NPS))])


%Radial averaging of the 2D-NPS, Version 1 (i.e. of the 2D-FFT of the noise image)
nps_measured_radagv = radialAverage(nps_measured, (NPS_roi_size+0)/2, (NPS_roi_size+0)/2, [0:1:NPS_roi_size/2-1]);
f_avg = linspace(0,1/px/2,length(nps_measured_radagv));

% [Zr, R] = radialavg(nps_measured,8);
% if DEBUG_PLOT_LEVEL >= 2
%     figure;plot(R,Zr);
% end

%% Summarize results as a structure
NPSresult.nps_measured = nps_measured;
NPSresult.f = f;
NPSresult.nps_measured_radagv = nps_measured_radagv;
NPSresult.f_avg = f_avg;
NPSresult.var_measured = var_measured;
NPSresult.var_NPS = var_NPS;


%% Save results in a txt-file
% ##if HOST_TYPE == 0
% ##  txtPath = strcat('Results\');%strcat(ImagePathInfo.PathResults);
% ##  txtName = strcat('Pat-',int2str(PatientNumber),'-Results-NPS'); %strcat(ImagePathInfo.FName,'-MTFarea','-',ROIname);
% ##  txtPathFnameFull = strcat(txtPath,txtName);
% ##
% ##  fileID = fopen(strcat(txtPathFnameFull,'.txt'),'w');
% ##  fprintf(fileID,'-------------------------------------------------------\n');
% ##  if ENV_TYPE == 0 %MATLAB
% ##    fprintf(fileID,'Results timestamp: %s\n',datetime);
% ##  end
% ##
% ##  fprintf(fileID,'Results for Dataset: %s  ## image %s (Patient Number used in Matlab: %d) \n',ImagePathInfo.SubFolderName, ImagePathInfo.FName, PatientNumber);
% ##  %fprintf(fileID,'ROI size= %3.0f x %3.0f pixel / ROI center positon (X,Y): (%3.0f , %3.0f) \n',ROI_SIZE, ROI_SIZE, ROI_X, ROI_Y);
% ##  fprintf(fileID,'Pixel variance measured with var() = %6.0f \n',var_measured);
% ##  fprintf(fileID,'Pixel variance measured with NPS() = %6.0f \n',var_NPS);
% ##
% ##  fprintf(fileID,'\n \n \n');
% ##  fclose(fileID);
% ##end


%% Plot results

%% PLOT - 1D NPS obtained from radial avaerging
if DEBUG_PLOT_LEVEL >= 1
    h=figure('Position', [100 100 700 700]);
    h1 = plot(f_avg(2:end),nps_measured_radagv(2:end),'x-');
    grid on
    xlabel('Frequency [mm^{-1}]'); 
    ylabel('NPS [mm^2]')
    xlim([0 1/px/2])
    ylim([0 max(nps_measured_radagv(:))])
    set(h1, 'LineWidth', 4);
    set(gca, 'Box', 'off' ); 
    set(gca, 'TickDir', 'out','LineWidth',2, 'TickLength',[0.02 0.08]);
    set(gca,'FontSize',12);

    %Export image as EPS and JPG and PNG
    imgPath = strcat('Images\');
    imgName = strcat('Pat-',int2str(PatientNumber),'-NPS1D-HomogenROI');
    imgPathFnameFull = strcat(imgPath,imgName);
    if ENV_TYPE == 0, hgexport(h,strcat(imgPathFnameFull,'.eps')), end
    saveas(h,strcat(imgPathFnameFull,'.png'));
    print(h,'-djpeg',strcat(imgPathFnameFull,'.jpg'));
end

%% PLOT - 2D NPS 
if DEBUG_PLOT_LEVEL >= 1
    h=figure('Position', [100 100 700 700]);
    imagesc(f,f,abs(nps_measured(:,:,1)))
    xlabel('Frequency [mm^{-1}]'); 
    ylabel('Frequency [mm^{-1}]');
    set(gca,'View',[-0 -90]) %Rotate the figure
    %set(gca,'visible','off') %Hide axes
    xlim([-1/px/2 1/px/2])
    ylim([-1/px/2 1/px/2])
    set(gca, 'Box', 'off' ); 
    set(gca, 'TickDir', 'out','LineWidth',2, 'TickLength',[0.02 0.08]);
    set(gca,'FontSize',12);
    
    %Export image as EPS and JPG and PNG
    imgPath = strcat('Images\');
    imgName = strcat('Pat-',int2str(PatientNumber),'-NPS2D-HomogenROI');
    imgPathFnameFull = strcat(imgPath,imgName);
    if ENV_TYPE == 0, hgexport(h,strcat(imgPathFnameFull,'.eps')), end
    saveas(h,strcat(imgPathFnameFull,'.png'));
    print(h,'-djpeg',strcat(imgPathFnameFull,'.jpg'));
end

return %End of this function